# Shared library for the "advanced" track: imagine_step8_advanced_a / b / c.  (Not run directly.)
# Design: ScienceQA image questions, ANSWER-FIRST format:  [question+choices] [visual tokens] "Answer:" -> letter
#   - the loss/accuracy is measured on the answer letter only (restricted to the valid choices)
#   - visual tokens come from a small trained "reader" (CLIP features -> LLM input embeddings)
import os, json, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from imagine_data import CACHE, CKPT, OUT, banner, set_seed, load_clip

LETTERS = "ABCDEFGH"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def amp(): return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda"))
def short(name): return name.split("/")[-1].lower()
def tag_of(a): return f"{short(a.model)}_{a.vis}" + (str(a.grid) if a.vis == "grid" else "")

# ----------------------------------------------------------------------------- data
def get_sqa_split(split, n_max):
    """ScienceQA image questions of one official split (train / validation / test); images saved to disk."""
    path = f"{CACHE}/adv_sqa_{split}_{n_max}.json"
    if os.path.exists(path):
        items = json.load(open(path))
        if all(os.path.exists(i["image_path"]) for i in items[:20]):
            print(f"[data] ScienceQA[{split}] cache hit: {len(items)} image questions", flush=True); return items
    from datasets import load_dataset
    folder = f"{CACHE}/adv_img/{split}"; os.makedirs(folder, exist_ok=True)
    ds = load_dataset("derek-thomas/ScienceQA", split=split, streaming=True)
    items, pbar = [], tqdm(total=n_max, desc=f"ScienceQA[{split}] image questions")
    for row in ds:
        if row.get("image") is None: continue
        img = row["image"].convert("RGB"); img.thumbnail((336, 336))
        p = f"{folder}/{len(items):05d}.jpg"; img.save(p, quality=92)
        items.append({"question": row["question"], "choices": list(row["choices"]), "answer": int(row["answer"]),
                      "hint": (row.get("hint") or "").strip(), "subject": row.get("subject", ""), "category": row.get("category", ""), "image_path": p})
        pbar.update(1)
        if len(items) >= n_max: break
    pbar.close(); json.dump(items, open(path, "w"))
    print(f"[data] ScienceQA[{split}]: {len(items)} image questions saved", flush=True)
    return items

def pad_square(im, fill=(255, 255, 255)):
    w, h = im.size; s = max(w, h); out = Image.new("RGB", (s, s), fill); out.paste(im, ((s - w) // 2, (s - h) // 2)); return out

@torch.no_grad()
def clip_features(paths, grid, tag, bs=32):
    """CLIP ViT-B/32: penultimate patch tokens pooled to grid x grid (LLaVA-style) + global 512-d embedding. Cached."""
    f = f"{CACHE}/adv_feats_{tag}_g{grid}.pt"
    if os.path.exists(f):
        d = torch.load(f)
        if d["n"] == len(paths): print(f"[data] CLIP features cache hit [{tag}]", flush=True); return d
    m, p, cdev = load_clip(); G, Gl = [], []
    for i in tqdm(range(0, len(paths), bs), desc=f"CLIP features [{tag}]"):
        ims = [pad_square(Image.open(x).convert("RGB")) for x in paths[i:i + bs]]
        px = p(images=ims, return_tensors="pt")["pixel_values"].to(cdev)
        vo = m.vision_model(pixel_values=px, output_hidden_states=True)
        pen = vo.hidden_states[-2][:, 1:, :]                       # [B,49,768] patch tokens
        B, N, C = pen.shape; s = int(round(N ** 0.5))
        g = F.adaptive_avg_pool2d(pen.transpose(1, 2).reshape(B, C, s, s), grid).flatten(2).transpose(1, 2)   # [B,grid*grid,768]
        G.append(g.half().cpu()); Gl.append(F.normalize(m.visual_projection(vo.pooler_output), dim=-1).half().cpu())
    d = {"grid": torch.cat(G), "glob": torch.cat(Gl), "n": len(paths)}; torch.save(d, f); return d

class VisSet:
    """Per-image visual feature tokens. mode 'global': 1 token x 512 | mode 'grid': grid*grid tokens x 768."""
    def __init__(self, feats, mode):
        self.mode = mode; self.x = feats["grid"] if mode == "grid" else feats["glob"].unsqueeze(1)
        self.n_tok, self.d_v = self.x.shape[1], self.x.shape[2]
    def get(self, idx): return self.x[torch.as_tensor(idx)].float()
    def target(self, idx): return F.layer_norm(self.get(idx), (self.d_v,))     # normalised regression target for imagination

def prepare_split(split, n, vis, grid, source="scienceqa"):
    if source == "spatial":
        from imagine_data import get_spatial_cot
        items = get_spatial_cot(split, n)
        tag = f"spatial_{split}{len(items)}"
    else:
        items = get_sqa_split(split, n); tag = f"{split}{len(items)}"
    feats = clip_features([i["image_path"] for i in items], grid, tag)
    return items, VisSet(feats, vis)

# ----------------------------------------------------------------------------- model parts
def load_lm(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[load] language model {name} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(dev).float(); model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    print(f"[load] d_model={model.config.hidden_size} layers={model.config.num_hidden_layers} params={sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    return tok, model

def attach_lora(model, r):
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
    pm.print_trainable_parameters(); return pm

def load_lora(model, path, trainable=False):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, path, is_trainable=trainable)

def lm_parts(pm, grad_ckpt=False):
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    if grad_ckpt: base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    return base.get_input_embeddings(), base.model, base.lm_head.weight

class VisualReader(nn.Module):
    """CLIP feature tokens -> LLM input embeddings (+ learned 'null' tokens used when no image is available)."""
    def __init__(self, d_v, n_tok, d_model, emb_rms):
        super().__init__()
        self.ln = nn.LayerNorm(d_v); self.mlp = nn.Sequential(nn.Linear(d_v, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.randn(n_tok, d_model) * 0.02); self.null = nn.Parameter(torch.randn(n_tok, d_model) * emb_rms)
        self.gamma = nn.Parameter(torch.tensor(float(emb_rms)))
    def forward(self, feats, null_mask=None):
        x = F.layer_norm(self.mlp(self.ln(feats)) + self.pos, (self.pos.size(-1),)) * self.gamma     # same scale as real token embeddings
        if null_mask is not None: x = torch.where(null_mask.view(-1, 1, 1), self.null.unsqueeze(0).expand_as(x).to(x.dtype), x)
        return x

class VisualImaginationHead(nn.Module):
    """text hidden state h (end of question) -> predicted visual feature tokens [n_tok, d_v]  (the 'Imagination Head')."""
    def __init__(self, d_model, d_v, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_v = n_tok, d_v; self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_tok * d_v))
    def forward(self, h): return self.net(self.ln(h.float())).view(-1, self.n_tok, self.d_v).float()

# ----------------------------------------------------------------------------- prompts + forward
def build_prompt(it):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
    ctx = f"Context: {it['hint'][:600]}\n" if it["hint"] else ""
    return f"Question: {it['question']}\n{ctx}Choices:\n{ch}\n"

def tokenize_items(tok, items, max_prompt=384):
    for it in items: it["pids"] = torch.tensor(tok(build_prompt(it), add_special_tokens=False).input_ids[-max_prompt:])

def answer_tokens(tok):
    ans = torch.tensor(tok("Answer:", add_special_tokens=False).input_ids)
    lid = torch.tensor([tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS])
    assert len(set(lid.tolist())) == len(LETTERS), "letter tokens are not distinct for this tokenizer"
    return ans, lid

def answer_logits(parts, items, vtok, ans_ids, letter_ids):
    """[B, 8] logits over answer letters (invalid choices masked). Sequence = prompt | visual tokens | 'Answer:'."""
    embed, body, head_w = parts; seqs = []
    for i, it in enumerate(items):
        pieces = [embed(it["pids"].to(dev))]
        if vtok is not None: pieces.append(vtok[i].to(pieces[0].dtype))
        pieces.append(embed(ans_ids.to(dev))); seqs.append(torch.cat(pieces, 0))
    lens = torch.tensor([s.size(0) for s in seqs], device=dev); L = int(lens.max())
    x = torch.stack([F.pad(s, (0, 0, 0, L - s.size(0))) for s in seqs])                  # right padding
    mask = (torch.arange(L, device=dev)[None, :] < lens[:, None]).long()
    h = body(inputs_embeds=x, attention_mask=mask).last_hidden_state[torch.arange(len(items), device=dev), lens - 1]
    logits = (h @ head_w[letter_ids.to(dev)].T).float()
    n = torch.tensor([len(it["choices"]) for it in items], device=dev)
    return logits.masked_fill(torch.arange(logits.size(1), device=dev)[None, :] >= n[:, None], -1e4)

@torch.no_grad()
def prompt_hidden(parts, items, bs=16, desc="prompt hidden"):
    """hidden state at the last question token = what the Imagination Head sees (a blank-slate student has only the question)."""
    embed, body, _ = parts; out = []
    for i in tqdm(range(0, len(items), bs), desc=desc, leave=False):
        seqs = [embed(it["pids"].to(dev)) for it in items[i:i + bs]]
        lens = torch.tensor([s.size(0) for s in seqs], device=dev); L = int(lens.max())
        x = torch.stack([F.pad(s, (0, 0, 0, L - s.size(0))) for s in seqs]); mask = (torch.arange(L, device=dev)[None, :] < lens[:, None]).long()
        with amp(): h = body(inputs_embeds=x, attention_mask=mask).last_hidden_state[torch.arange(len(seqs), device=dev), lens - 1]
        out.append(h.float().cpu())
    return torch.cat(out)

# ----------------------------------------------------------------------------- evaluation + statistics
@torch.no_grad()
def eval_conditions(parts, items, ans_ids, letter_ids, cond_fns, bs=16, desc="eval"):
    """cond_fns: {name: fn(idx_list) -> visual tokens [B,T,d]}. Returns {name: {'ce': [...], 'ok': [...]}} per example."""
    res = {k: {"ce": [], "ok": []} for k in cond_fns}
    for i in tqdm(range(0, len(items), bs), desc=desc, leave=False):
        idx = list(range(i, min(i + bs, len(items)))); batch = [items[j] for j in idx]
        gold = torch.tensor([it["answer"] for it in batch], device=dev)
        for name, fn in cond_fns.items():
            with amp(): lg = answer_logits(parts, batch, fn(idx), ans_ids, letter_ids)
            res[name]["ce"] += F.cross_entropy(lg, gold, reduction="none").tolist(); res[name]["ok"] += (lg.argmax(1) == gold).float().tolist()
    return res

def boot_ci(diff, n=4000, seed=0):
    d = torch.tensor(diff, dtype=torch.float32); g = torch.Generator().manual_seed(seed)
    means = d[torch.randint(0, len(d), (n, len(d)), generator=g)].mean(1)
    return d.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()

def summarize(res, items, title):
    chance_acc = sum(1 / len(it["choices"]) for it in items) / len(items); chance_ce = sum(math.log(len(it["choices"])) for it in items) / len(items)
    print(f"\n=== {title} (n={len(items)}; chance: acc={chance_acc*100:.1f}% CE={chance_ce:.3f}) ===")
    for k, v in res.items(): print(f"  {k:13s} acc={sum(v['ok'])/len(v['ok'])*100:5.1f}%   CE={sum(v['ce'])/len(v['ce']):.4f}")

def contrast(res, label, better, worse):
    """Paired bootstrap on (worse - better) CE and (better - worse) accuracy. Positive = 'better' really is better."""
    m, lo, hi = boot_ci([w - b for w, b in zip(res[worse]["ce"], res[better]["ce"])])
    am, alo, ahi = boot_ci([b - w for w, b in zip(res[worse]["ok"], res[better]["ok"])])
    verdict = "SUPPORTED" if lo > 0 else ("OPPOSITE" if hi < 0 else "not distinguishable from 0")
    print(f"  {label:52s} CE {m:+.4f} [{lo:+.4f},{hi:+.4f}] | acc {am*100:+.2f} [{alo*100:+.2f},{ahi*100:+.2f}] pts -> {verdict}")
    return m, lo, hi

def cosine_lr(opt, total, warmup=20):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
