# Step 8-advanced-A0  (alignment stage, LLaVA-style): teach the reader to speak "image" to the LLM using CAPTIONS.
# Why: in Phase A the reader only got an answer-letter signal from 4000 questions -> too weak to learn to extract anything from the image.
#      Captions give a rich, dense signal (every caption token depends on the image). The LLM stays FROZEN here; only the reader trains.
# Built-in sanity check: caption loss with the TRUE image must be clearly lower than with a WRONG image or blank tokens.
#      If that gap is ~0, the visual path itself is broken/too weak (a useful diagnosis in its own right).
# Run:  python imagine_step8_advanced_a0.py --vis grid --grid 3          (then re-run imagine_step8_advanced_a.py, it picks this reader up automatically)
import argparse, json, os, random, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_adv_lib import *
from imagine_data import get_coco

def caption_ce(parts, vtok, prompt_ids, caps):
    """per-example mean CE of the caption tokens given [visual tokens][prompt]. caps: list of 1D token tensors."""
    embed, body, head_w = parts; seqs, spans = [], []
    pe = embed(prompt_ids.to(dev))
    for i, c in enumerate(caps):
        seqs.append(torch.cat([vtok[i].to(pe.dtype), pe, embed(c.to(dev))], 0)); spans.append((vtok.size(1) + len(prompt_ids), len(c)))
    lens = torch.tensor([s.size(0) for s in seqs], device=dev); L = int(lens.max())
    x = torch.stack([F.pad(s, (0, 0, 0, L - s.size(0))) for s in seqs]); mask = (torch.arange(L, device=dev)[None, :] < lens[:, None]).long()
    h = body(inputs_embeds=x, attention_mask=mask).last_hidden_state
    hs = torch.cat([h[i, s - 1:s - 1 + n] for i, (s, n) in enumerate(spans)]); tg = torch.cat([c.to(dev) for c in caps])
    ce = F.cross_entropy((hs @ head_w.T).float(), tg, reduction="none")
    return torch.stack([t.mean() for t in torch.split(ce, [n for _, n in spans])])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--vis", default="global", choices=["global", "grid"]); ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--n_img", type=int, default=20000); ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--p_null", type=float, default=0.1)
    a = ap.parse_args(); set_seed(0); out = f"{CKPT}/adv_a0_{tag_of(a)}"; os.makedirs(out, exist_ok=True)

    banner("A0-1: COCO captions + CLIP features")
    n = a.n_img; meta = None
    while meta is None:
        try: meta = get_coco(n)
        except RuntimeError as e:
            if n <= 2000: raise
            n //= 2; print(f"[data] retrying with {n} images ({e})")
    feats = clip_features([m["path"] for m in meta], a.grid, f"coco{len(meta)}"); vis = VisSet(feats, a.vis)
    nv = a.n_val; ntr = len(meta) - nv; print(f"images: {len(meta)} (train {ntr}, val {nv}) | visual tokens per image: {vis.n_tok} x {vis.d_v}")

    banner("A0-2: frozen LLM + trainable reader")
    tok, model = load_lm(a.model); model.eval(); d_model = model.config.hidden_size; parts = lm_parts(model)
    emb_rms = model.get_input_embeddings().weight.float().pow(2).mean().sqrt().item()
    reader = VisualReader(vis.d_v, vis.n_tok, d_model, emb_rms).to(dev)
    prompt_ids = torch.tensor(tok("A short caption of the image:", add_special_tokens=False).input_ids)
    enc = lambda c: torch.tensor(tok(" " + c.strip(), add_special_tokens=False).input_ids[:40] + [tok.eos_token_id])
    print("example:", meta[0]["captions"][0])

    @torch.no_grad()
    def validate(title):
        reader.eval(); off = nv // 2; vi = list(range(ntr, len(meta))); res = {"true": [], "wrong": [], "blank": []}
        for i in tqdm(range(0, nv, 32), desc=title, leave=False):
            b = list(range(i, min(i + 32, nv))); caps = [enc(meta[ntr + j]["captions"][0]) for j in b]
            with amp():
                res["true"] += caption_ce(parts, reader(vis.get([vi[j] for j in b]).to(dev)), prompt_ids, caps).tolist()
                res["wrong"] += caption_ce(parts, reader(vis.get([vi[(j + off) % nv] for j in b]).to(dev)), prompt_ids, caps).tolist()
                res["blank"] += caption_ce(parts, reader(vis.get([vi[j] for j in b]).to(dev), torch.ones(len(b), dtype=torch.bool, device=dev)), prompt_ids, caps).tolist()
        print(f"\n=== {title}: caption CE on {nv} held-out COCO images ===  " + " | ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in res.items()))
        for lab, w in (("true image beats WRONG image", "wrong"), ("true image beats BLANK tokens", "blank")):
            m, lo, hi = boot_ci([x - y for x, y in zip(res[w], res["true"])])
            print(f"  {lab:32s} CE gap {m:+.3f} [{lo:+.3f},{hi:+.3f}] -> {'SUPPORTED' if lo > 0 else 'not distinguishable from 0'}")
        torch.cuda.empty_cache(); return m, lo

    banner("A0-3: BEFORE training"); validate("before")
    banner("A0-4: training the reader (LLM frozen)")
    opt = torch.optim.AdamW(reader.parameters(), lr=a.lr, weight_decay=0.01); steps = ntr // a.bs; sched = cosine_lr(opt, steps * a.epochs, warmup=100)
    for ep in range(a.epochs):
        reader.train(); order = list(range(ntr)); random.shuffle(order); log = []; pbar = tqdm(range(steps), desc=f"epoch {ep+1}/{a.epochs}")
        for s in pbar:
            idx = order[s * a.bs:(s + 1) * a.bs]; caps = [enc(random.choice(meta[j]["captions"])) for j in idx]; null = torch.rand(len(idx), device=dev) < a.p_null
            with amp(): loss = caption_ce(parts, reader(vis.get(idx).to(dev), null), prompt_ids, caps).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0); opt.step(); sched.step(); log.append(loss.item())
            pbar.set_postfix(loss=f"{sum(log[-50:])/len(log[-50:]):.3f}")
        gap, lo = validate(f"epoch {ep+1}")
        torch.save({"reader": reader.state_dict(), "epoch": ep + 1, "d_v": vis.d_v, "n_tok": vis.n_tok, "emb_rms": emb_rms}, f"{out}/reader.pt")
    print("\nA0", "PASSED: the LLM now uses visual tokens (caption loss depends on the image). Re-run imagine_step8_advanced_a.py -> it loads this reader." if lo > 0 else "NOT PASSED: the visual path is not carrying image information; tell me the numbers.")
    print(f"saved {out}/reader.pt\nSTEP 8-ADVANCED-A0 OK")

if __name__ == "__main__":
    main()
