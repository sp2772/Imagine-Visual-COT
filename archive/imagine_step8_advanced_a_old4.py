# Step 8-advanced-A  (Phase A: "the student learns to SEE")
# Goal: make the image NECESSARY and measurable. Train an answer-first reader on ScienceQA IMAGE questions so that
#       true image tokens beat wrong-image tokens and blank tokens. Only when this is SUPPORTED does it make sense to ask
#       whether the model can imagine those tokens itself (Phase B).
# Model: Qwen2.5-0.5B + LoRA (bf16 autocast, gradient checkpointing -> fits a 6GB GPU; slower but fine).
# Run (do both and compare):
#   python imagine_step8_advanced_a.py --vis global                 # 1 CLIP vector, like before (baseline)
#   python imagine_step8_advanced_a.py --vis grid --grid 3          # 3x3 grid of CLIP patch tokens (more visual detail)
# If you get CUDA out-of-memory: --micro_bs 4 --accum 4     Stop/resume: add --resume
import argparse, json, os, random, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_adv_lib import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--vis", default="global", choices=["global", "grid"]); ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--data", default="scienceqa", choices=["scienceqa", "spatial"], help="spatial = ball-in-box dataset from imagine_build_spatial_cot.py")
    ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--micro_bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--lr_lora", type=float, default=2e-4); ap.add_argument("--lr_reader", type=float, default=5e-4)
    ap.add_argument("--p_null", type=float, default=0.15, help="prob. of replacing the image by learned blank tokens during training")
    ap.add_argument("--w_contrast", type=float, default=0.0, help="weight of the wrong-image margin loss (forces the model to USE the image; ~2x slower)")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--no_init", action="store_true", help="do not load the caption-aligned reader from step A0")
    ap.add_argument("--no_grad_ckpt", action="store_true"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); set_seed(0)
    out = f"{CKPT}/adv_a_{tag_of(a)}" + ("" if a.data == "scienceqa" else f"_{a.data}")
    os.makedirs(out, exist_ok=True)

    banner(f"A1: data  ({a.data}, vis={a.vis})")
    tr, vtr = prepare_split("train", a.n_train, a.vis, a.grid, source=a.data)
    va, vva = prepare_split("validation", a.n_val, a.vis, a.grid, source=a.data)
    print(f"train {len(tr)} | val {len(va)} | visual tokens per question: {vtr.n_tok} x {vtr.d_v}")

    banner("A2: model")
    tok, model = load_lm(a.model); d_model = model.config.hidden_size; start_epoch = 0
    emb_rms = model.get_input_embeddings().weight.float().pow(2).mean().sqrt().item()
    reader = VisualReader(vtr.d_v, vtr.n_tok, d_model, emb_rms).to(dev)
    init = f"{CKPT}/adv_a0_{tag_of(a)}/reader.pt"
    if os.path.exists(init) and not a.no_init: reader.load_state_dict(torch.load(init)["reader"]); print("reader initialised from the caption-alignment stage A0:", init)
    if a.resume and os.path.exists(f"{out}/meta.json"):
        pm = load_lora(model, f"{out}/lora", trainable=True); ck = torch.load(f"{out}/reader.pt"); reader.load_state_dict(ck["reader"]); start_epoch = ck["epoch"]
        print(f"resumed from epoch {start_epoch}")
    else: pm = attach_lora(model, a.r)
    parts = lm_parts(pm, grad_ckpt=not a.no_grad_ckpt); ans_ids, letter_ids = answer_tokens(tok); tokenize_items(tok, tr); tokenize_items(tok, va)
    print("prompt tokens (median):", sorted(len(i["pids"]) for i in tr)[len(tr) // 2], "| answer tokens:", ans_ids.tolist(), "| letter tokens:", letter_ids[:4].tolist())

    off = len(va) // 2
    conds = {"oracle": lambda idx: reader(vva.get(idx).to(dev)),                                         # true image
             "wrong": lambda idx: reader(vva.get([(j + off) % len(va) for j in idx]).to(dev)),            # another question's image
             "const": lambda idx: reader(vva.get(idx).to(dev), torch.ones(len(idx), dtype=torch.bool, device=dev))}   # blank tokens
    def validate(title):
        pm.eval(); reader.eval(); res = eval_conditions(parts, va, ans_ids, letter_ids, conds, desc=title); summarize(res, va, title)
        contrast(res, "image content is usable   (oracle vs wrong)", "oracle", "wrong")
        contrast(res, "image helps vs blank      (oracle vs const)", "oracle", "const")
        contrast(res, "any-image-token effect    (wrong vs const)", "wrong", "const")
        torch.cuda.empty_cache(); return res

    if start_epoch == 0: banner("A3: BEFORE training (text-only prior of the base model)"); validate("before")
    banner("A4: training")
    params = list(reader.parameters()) + [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": list(reader.parameters()), "lr": a.lr_reader}, {"params": [p for p in pm.parameters() if p.requires_grad], "lr": a.lr_lora}], weight_decay=0.01)
    per_step = a.micro_bs * a.accum; steps = len(tr) // per_step; sched = cosine_lr(opt, steps * a.epochs)
    for _ in range(start_epoch * steps): sched.step()
    for ep in range(start_epoch, a.epochs):
        pm.train(); reader.train(); order = list(range(len(tr))); random.shuffle(order); la = []; aa = []
        pbar = tqdm(range(steps), desc=f"epoch {ep+1}/{a.epochs}")
        for s in pbar:
            opt.zero_grad()
            for k in range(a.accum):
                idx = order[(s * a.accum + k) * a.micro_bs:(s * a.accum + k + 1) * a.micro_bs]; batch = [tr[j] for j in idx]
                gold = torch.tensor([it["answer"] for it in batch], device=dev); null = torch.rand(len(idx), device=dev) < a.p_null
                with amp(): lg = answer_logits(parts, batch, reader(vtr.get(idx).to(dev), null), ans_ids, letter_ids)
                ce = F.cross_entropy(lg, gold, reduction="none"); loss = ce.mean()
                if a.w_contrast > 0:   # the TRUE image must make the right answer more likely than another question's image does
                    idx_w = idx[1:] + idx[:1]
                    with amp(): lg_w = answer_logits(parts, batch, reader(vtr.get(idx_w).to(dev)), ans_ids, letter_ids)
                    ce_w = F.cross_entropy(lg_w, gold, reduction="none")
                    loss = loss + a.w_contrast * (F.relu(a.margin - (ce_w - ce)) * (~null).float()).mean()
                (loss / a.accum).backward(); la.append(ce.mean().item()); aa.append((lg.argmax(1) == gold).float().mean().item())
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            pbar.set_postfix(loss=f"{sum(la[-20:])/len(la[-20:]):.3f}", acc=f"{sum(aa[-20:])/len(aa[-20:]):.3f}")
        validate(f"epoch {ep+1}")
        pm.save_pretrained(f"{out}/lora"); torch.save({"reader": reader.state_dict(), "epoch": ep + 1, "d_v": vtr.d_v, "n_tok": vtr.n_tok, "emb_rms": emb_rms}, f"{out}/reader.pt")
        json.dump({"epoch": ep + 1, "args": vars(a)}, open(f"{out}/meta.json", "w"))
    banner("A5: final validation")
    res = validate("final")
    m, lo, hi = contrast(res, "image content is usable   (oracle vs wrong)", "oracle", "wrong")
    print("\nPHASE A", "PASSED: the reader uses image content -> run Phase B." if lo > 0 else "NOT PASSED yet: try --vis grid, more epochs (--epochs 8), or a larger --n_train before Phase B.")
    print(f"saved {out}\nSTEP 8-ADVANCED-A OK")

if __name__ == "__main__":
    main()
