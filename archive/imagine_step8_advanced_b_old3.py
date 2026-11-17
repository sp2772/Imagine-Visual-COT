# Step 8-advanced-B  (Phase B: "the student learns to IMAGINE")
# The reader from Phase A (LoRA + visual reader) is FROZEN. We train only the Imagination Head:
#     question text -> hidden state h -> predicted visual tokens -> frozen reader -> frozen LLM -> answer
# Teacher = the true image tokens (never given at test time). Losses (all use the image only as a teaching signal):
#     feat : imagined tokens match the true CLIP tokens (cosine + small MSE)
#     nce  : contrastive semantic-preservation across the batch (imagination must stay specific to ITS question)
#     ans  : answer CE through the frozen reader        kd : KL(answer distribution with true image || with imagination)
# Run:  python imagine_step8_advanced_b.py --vis grid --grid 3     (use the same --vis/--grid/--model as Phase A)
import argparse, json, os, random, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_adv_lib import *

def sym_nce(a, b, t=0.07):
    a = F.normalize(a - a.mean(0, keepdim=True), dim=-1); b = F.normalize(b - b.mean(0, keepdim=True), dim=-1)   # centre: remove the shared 'average image'
    lg = a @ b.T / t; y = torch.arange(len(a), device=a.device); return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--vis", default="global", choices=["global", "grid"]); ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--micro_bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--w_feat", type=float, default=1.0); ap.add_argument("--w_nce", type=float, default=0.5); ap.add_argument("--w_ans", type=float, default=1.0); ap.add_argument("--w_kd", type=float, default=1.0)
    ap.add_argument("--data", default="scienceqa", choices=["scienceqa", "spatial", "spatial2"]); ap.add_argument("--track", default="all", choices=["all", "imagine", "see"], help="imagine = only items whose TEXT specifies the scene (the only ones where imagining from text can help)"); ap.add_argument("--no_grad_ckpt", action="store_true")
    a = ap.parse_args(); set_seed(0); src = f"{CKPT}/adv_a_{tag_of(a)}{suffix_of(a)}"; out = f"{CKPT}/adv_b_{tag_of(a)}{suffix_of(a)}{trk_suffix(a)}"; os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{src}/reader.pt"): raise SystemExit(f"Phase A checkpoint not found at {src}. Run imagine_step8_advanced_a.py with the same --model/--vis/--grid first.")

    banner("B1: data + frozen reader from Phase A")
    tr, vtr = prepare_split("train", a.n_train, a.vis, a.grid, source=a.data, track=a.track); va, vva = prepare_split("validation", a.n_val, a.vis, a.grid, source=a.data, track=a.track)
    tok, model = load_lm(a.model); d_model = model.config.hidden_size
    ck = torch.load(f"{src}/reader.pt"); reader = VisualReader(ck["d_v"], ck["n_tok"], d_model, ck["emb_rms"]).to(dev); reader.load_state_dict(ck["reader"])
    pm = load_lora(model, f"{src}/lora", trainable=False); parts = lm_parts(pm, grad_ckpt=not a.no_grad_ckpt)
    for p in list(reader.parameters()) + list(pm.parameters()): p.requires_grad_(False)
    ans_ids, letter_ids = answer_tokens(tok); tokenize_items(tok, tr); tokenize_items(tok, va); pm.eval(); reader.eval()
    Htr = prompt_hidden(parts, tr, desc="hidden (train)"); Hva = prompt_hidden(parts, va, desc="hidden (val)"); print("question hidden states:", tuple(Htr.shape))

    head = VisualImaginationHead(d_model, vtr.d_v, vtr.n_tok).to(dev)
    print("imagination head params: %.1fM" % (sum(p.numel() for p in head.parameters()) / 1e6))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01); per_step = a.micro_bs * a.accum; steps = len(tr) // per_step; sched = cosine_lr(opt, steps * a.epochs)

    off = len(va) // 2
    def imagine(idx, H): return reader(head(H[torch.as_tensor(idx)].to(dev)))
    conds = {"oracle": lambda idx: reader(vva.get(idx).to(dev)),
             "own": lambda idx: imagine(idx, Hva),                                                      # BLANK SLATE: question text only
             "own_shuffled": lambda idx: imagine([(j + off) % len(va) for j in idx], Hva),              # imagination of ANOTHER question
             "wrong": lambda idx: reader(vva.get([(j + off) % len(va) for j in idx]).to(dev)),
             "const": lambda idx: reader(vva.get(idx).to(dev), torch.ones(len(idx), dtype=torch.bool, device=dev))}
    @torch.no_grad()
    def validate(title):
        head.eval(); res = eval_conditions(parts, va, ans_ids, letter_ids, conds, desc=title); summarize(res, va, title); breakdown(res, va, "track"); breakdown(res, va, "family")
        print("  -- does the student's OWN imagination (question text only) help?")
        contrast(res, "own imagination vs blank        (const - own)", "own", "const")
        contrast(res, "own is example-specific  (own_shuffled - own)", "own", "own_shuffled")
        contrast(res, "own vs a random real image      (wrong - own)", "own", "wrong")
        contrast(res, "headroom: true image vs own     (own - oracle)", "oracle", "own")
        use = contrast(res, "[reference] image content usable (wrong - oracle)", "oracle", "wrong")
        ce = {k: sum(v["ce"]) / len(v["ce"]) for k, v in res.items()}
        if use[1] > 0 and ce["const"] > ce["oracle"]: print(f"  recovery of the teacher's benefit (CE): {(ce['const'] - ce['own']) / (ce['const'] - ce['oracle']) * 100:.0f}%  (own vs blank, relative to true image vs blank)")
        Z = F.normalize(head(Hva.to(dev)).mean(1) - head(Hva.to(dev)).mean((0, 1)), dim=-1); T = F.normalize(vva.target(list(range(len(va)))).to(dev).mean(1) - vva.target(list(range(len(va)))).to(dev).mean((0, 1)), dim=-1)
        sim = Z @ T.T; top1 = (sim.argmax(1) == torch.arange(len(va), device=dev)).float().mean().item(); top5 = (sim.topk(5, 1).indices == torch.arange(len(va), device=dev)[:, None]).any(1).float().mean().item()
        print(f"  imagined vs true image retrieval among {len(va)} val images: top1={top1:.3f} top5={top5:.3f} (chance {1/len(va):.3f} / {5/len(va):.3f})")
        head.train(); torch.cuda.empty_cache(); return res

    banner("B2: BEFORE training (random head)"); validate("before")
    banner("B3: training the Imagination Head")
    for ep in range(a.epochs):
        head.train(); pm.train(); order = list(range(len(tr))); random.shuffle(order); logs = {k: [] for k in ("feat", "nce", "ans", "kd", "acc")}
        pbar = tqdm(range(steps), desc=f"epoch {ep+1}/{a.epochs}")
        for s in pbar:
            opt.zero_grad()
            for k in range(a.accum):
                idx = order[(s * a.accum + k) * a.micro_bs:(s * a.accum + k + 1) * a.micro_bs]; batch = [tr[j] for j in idx]; gold = torch.tensor([it["answer"] for it in batch], device=dev)
                tgt = vtr.target(idx).to(dev)
                with amp():
                    imag = head(Htr[torch.as_tensor(idx)].to(dev)); lg_own = answer_logits(parts, batch, reader(imag), ans_ids, letter_ids)
                    with torch.no_grad(): lg_or = answer_logits(parts, batch, reader(vtr.get(idx).to(dev)), ans_ids, letter_ids)
                l_feat = (1 - F.cosine_similarity(imag, tgt, dim=-1)).mean() + 0.1 * F.mse_loss(imag, tgt)
                l_nce = sym_nce(imag.mean(1), tgt.mean(1)); l_ans = F.cross_entropy(lg_own, gold)
                l_kd = F.kl_div(F.log_softmax(lg_own, -1), F.softmax(lg_or, -1), reduction="batchmean")
                loss = a.w_feat * l_feat + a.w_nce * l_nce + a.w_ans * l_ans + a.w_kd * l_kd; (loss / a.accum).backward()
                for n_, v_ in (("feat", l_feat), ("nce", l_nce), ("ans", l_ans), ("kd", l_kd)): logs[n_].append(v_.item())
                logs["acc"].append((lg_own.argmax(1) == gold).float().mean().item())
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
            pbar.set_postfix({k: f"{sum(v[-20:])/len(v[-20:]):.3f}" for k, v in logs.items()})
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs: validate(f"epoch {ep+1}")
        torch.save({"head": head.state_dict(), "epoch": ep + 1, "d_v": vtr.d_v, "n_tok": vtr.n_tok}, f"{out}/head.pt")
    banner("B4: final"); res = validate("final")
    print("\nHOW TO READ: the key lines are 'own imagination vs blank' and 'own is example-specific'. SUPPORTED means the CI excludes 0, i.e. unlikely to be noise.")
    print("If Phase A did not pass, these can only show a soft-prompt effect. Final held-out numbers come from Phase C (test split).")
    print(f"saved {out}/head.pt\nSTEP 8-ADVANCED-B OK")

if __name__ == "__main__":
    main()
