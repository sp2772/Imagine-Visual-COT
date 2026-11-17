# Step 8 (Stages 2/3 combined, LLM frozen): joint training of head + gate + feedback projector on ScienceQA chains-of-thought.
# Losses: LM (text) + align (z_t vs CLIP text of step t, InfoNCE) + img (z vs CLIP image) + div (anti-collapse) + gate (BCE).
# NOTE: Visual-CoT is skipped (huge, images scattered over several source datasets); ScienceQA image questions stand in.
# Run: python imagine_step8.py --n_train 400 --n_val 80 --epochs 2 --bs 8
import argparse, os, random, time, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, make_segs, save_chain_parts
from imagine_data import *

LETTERS = "ABCDEFGH"
W = dict(align=1.0, img=0.5, div=0.5, gate=0.5, margin=0.8)

def amp(enabled): return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)

def build_prompt(it):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
    ctx = f"Context: {it['hint'][:400]}\n" if it["hint"] else ""
    return f"Question: {it['question']}\n{ctx}Choices:\n{ch}\nLet's think step by step.\n"

def prepare(items, tok):
    data, step_texts = [], []
    for it in items:
        steps = split_steps(it["solution"], 4)
        if len(steps) < 2: continue
        texts = [build_prompt(it)] + [f"Step {i+1}: {s}\n" for i, s in enumerate(steps)] + [f"Answer: {LETTERS[it['answer']]}\n"]
        data.append({"it": it, "texts": texts, "n_steps": len(steps)}); step_texts += steps
    print(f"[prep] {len(data)} usable examples, {len(step_texts)} reasoning steps", flush=True)
    emb = clip_text_embs(step_texts); k = 0
    for d in data: d["step_emb"] = emb[k:k + d["n_steps"]]; k += d["n_steps"]
    imgs = [d for d in data if d["it"]["has_image"]]
    if imgs:
        for d, e in zip(imgs, clip_image_embs([d["it"]["image_path"] for d in imgs])): d["img_emb"] = e
    for d in data:
        d["segs"] = make_segs(tok, d["texts"])
        d["masks"] = [torch.zeros(len(d["segs"][0]), dtype=torch.bool)] + [torch.ones(len(s), dtype=torch.bool) for s in d["segs"][1:]]
    return data

def load_data(n_train, n_val, tok):
    half = (n_train + n_val) // 2
    items = get_scienceqa(n_img=half, n_noimg=half, min_steps=2)
    random.Random(0).shuffle(items)
    data = prepare(items, tok)
    return data[n_val:], data[:n_val]

def cons_cos(z): return F.cosine_similarity(z[1:], z[:-1], dim=-1)

def batch_losses(chain, batch, use_amp):
    Z, T, lm, div, gate, imgl = [], [], [], [], [], []
    for d in batch:
        with amp(use_amp): r = chain(d["segs"], d["masks"])
        z, n = r["z"], d["n_steps"]
        Z.append(z[1:1 + n]); T.append(d["step_emb"].to(z.device)); lm.append(r["lm_loss"])
        div.append(F.relu(cons_cos(z) - W["margin"]).mean())
        gate.append(F.binary_cross_entropy(r["g"].clamp(1e-5, 1 - 1e-5), torch.full_like(r["g"], float(d["it"]["has_image"]))))
        if d["it"]["has_image"]:
            zm = F.normalize(z[1:1 + n].mean(0), dim=-1); imgl.append(1 - (zm * d["img_emb"].to(z.device)).sum())
    Z, T = torch.cat(Z), torch.cat(T)
    logits = (Z @ T.T) / 0.07; tgt = torch.arange(len(Z), device=Z.device)
    align = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
    parts = {"lm": torch.stack(lm).mean(), "align": align, "div": torch.stack(div).mean(), "gate": torch.stack(gate).mean(),
             "img": torch.stack(imgl).mean() if imgl else torch.zeros((), device=Z.device)}
    total = parts["lm"] + W["align"] * align + W["img"] * parts["img"] + W["div"] * parts["div"] + W["gate"] * parts["gate"]
    return total, {k: v.item() for k, v in parts.items()}

@torch.no_grad()
def evaluate(chain, data, use_amp=False):
    chain.eval(); lm_f = lm_c = 0.0; cos, ok, Z, T = [], [], [], []
    for d in tqdm(data, desc="eval", leave=False):
        with amp(use_amp):
            r = chain(d["segs"], d["masks"], mode="feedback"); rc = chain(d["segs"], d["masks"], mode="constant")
        lm_f += r["lm_loss"].item(); lm_c += rc["lm_loss"].item()
        cos.append(cons_cos(r["z"]).mean().item()); n = d["n_steps"]
        Z.append(r["z"][1:1 + n]); T.append(d["step_emb"].to(r["z"].device))
        ok.append(((r["g"] > 0.5).float() == float(d["it"]["has_image"])).float().mean().item())
    Z, T = torch.cat(Z), torch.cat(T); top1 = ((Z @ T.T).argmax(1) == torch.arange(len(Z), device=Z.device)).float().mean().item()
    n = len(data)
    return {"lm_feedback": lm_f / n, "lm_constant": lm_c / n, "imagination_benefit": (lm_c - lm_f) / n,
            "mean_cos_consecutive_z": sum(cos) / n, "align_top1": top1, "align_chance": 1 / len(Z), "gate_acc": sum(ok) / n}

def show(tag, ev): print(f"[{tag}] " + " | ".join(f"{k}={v:.4f}" for k, v in ev.items()), flush=True)

def train_loop(chain, groups, train, val, epochs, bs, use_amp, tag):
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    for ep in range(epochs):
        random.shuffle(train); chain.train()
        pbar = tqdm(range(0, len(train) - bs + 1, bs), desc=f"{tag} epoch {ep+1}/{epochs}")
        for i in pbar:
            opt.zero_grad()
            loss, parts = batch_losses(chain, train[i:i + bs], use_amp)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()})
        show(f"{tag} val after epoch {ep+1}", evaluate(chain, val, use_amp))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args(); set_seed(0)
    banner("8a: data (ScienceQA with chain-of-thought solutions)")
    tok, model = load_backbone(); d = model.config.hidden_size
    train, val = load_data(a.n_train, a.n_val, tok)
    ex = train[0]; print(f"train {len(train)} | val {len(val)}\nexample segments:"); [print("   ", repr(t[:110])) for t in ex["texts"]]
    banner("8b: model")
    chain = FeedbackChain(model, d).to(device)
    ck = f"{CKPT}/head_step6.pt" if os.path.exists(f"{CKPT}/head_step6.pt") else f"{CKPT}/head_step5.pt"
    if os.path.exists(ck): chain.head.load_state_dict(torch.load(ck, map_location=device)["head"]); print("loaded head from", ck)
    else: print("WARNING: no head checkpoint; run steps 5/6 first")
    groups = [{"params": list(chain.head.parameters()), "lr": 2e-4}, {"params": list(chain.proj.parameters()) + [chain.img_start], "lr": 1e-3}]
    print("trainable params:", sum(p.numel() for g in groups for p in g["params"]), "(LLM frozen)")
    banner("8c: baseline eval before training"); ev0 = evaluate(chain, val); show("before", ev0)
    banner("8d: training"); train_loop(chain, groups, train, val, a.epochs, a.bs, False, "stage8")
    banner("8e: summary"); ev1 = evaluate(chain, val); show("after ", ev1)
    print(f"\nKEY CHECKS: align_top1 {ev0['align_top1']:.3f}->{ev1['align_top1']:.3f} (chance {ev1['align_chance']:.3f}) | "
          f"consecutive cos {ev0['mean_cos_consecutive_z']:.3f}->{ev1['mean_cos_consecutive_z']:.3f} (lower = less collapse) | "
          f"imagination_benefit {ev1['imagination_benefit']:.4f} (>0 means feedback lowers the LM loss vs a constant token)")
    save_chain_parts(chain, f"{CKPT}/chain_step8.pt"); print(f"saved {CKPT}/chain_step8.pt\nSTEP 8 OK")

if __name__ == "__main__":
    main()
