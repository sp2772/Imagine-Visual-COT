# Step 5 (Stage 1): teach the Imagination Head to map text hidden states -> CLIP image embeddings (COCO captions).
# Run: python imagine_step5.py            (defaults: 2000 images, 40 epochs)
#      python imagine_step5.py --n 4000 --epochs 60
import argparse, math, os, time, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, batch_hidden, device
from imagine_step2 import ImaginationHead
from imagine_data import *

def retrieval_metrics(z, img):
    sim = z @ img.T                                         # diagonal = correct pair
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1)          # 0 means rank 1
    out = {f"top{k}": (ranks < k).float().mean().item() for k in (1, 5, 10)}
    out["median_rank"] = ranks.float().median().item() + 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pool", default="last", choices=["last", "mean"], help="diagnostic knob; chain uses 'last'")
    a = ap.parse_args(); set_seed(0)

    banner("5a: download COCO subset + CLIP image embeddings")
    meta = get_coco(a.n); N = len(meta); paths = [m["path"] for m in meta]
    print(f"images: {N} | example caption: {meta[0]['captions'][0]!r}")
    ep = f"{CACHE}/coco_clip_embs.pt"
    cached = torch.load(ep) if os.path.exists(ep) else None
    if cached is not None and cached["paths"] == paths: img = cached["embs"]; print("[data] CLIP image embs from cache")
    else:
        img = clip_image_embs(paths)
        torch.save({"paths": paths, "embs": img, "captions": [m["captions"] for m in meta]}, ep)
    print("image embs:", tuple(img.shape), "| mean norm: %.3f (expect 1.0)" % img.norm(dim=-1).mean())

    banner("5b: LLM hidden states of captions")
    tok, model = load_backbone()
    t0 = [m["captions"][0].strip() + "\n" for m in meta]
    t1 = [(m["captions"][1] if len(m["captions"]) > 1 else m["captions"][0]).strip() + "\n" for m in meta]
    H0 = batch_hidden(tok, model, t0, pool=a.pool, desc="hidden (caption 1)").to(device)
    H1 = batch_hidden(tok, model, t1, pool=a.pool, desc="hidden (caption 2)").to(device)
    img = img.to(device); d = H0.size(1); del model
    print("H0:", tuple(H0.shape))

    banner("5c: train head (InfoNCE: text hidden -> image embedding)")
    n_val = max(100, N // 10); tr = torch.arange(0, N - n_val); va = torch.arange(N - n_val, N).to(device)
    print(f"train images: {len(tr)} | val images: {n_val} | chance top1={1/n_val:.3f} top5={5/n_val:.3f}")
    head = ImaginationHead(d).to(device)
    scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    params = [p for n, p in head.named_parameters() if not n.startswith("W1")] + [scale]   # gate is trained in step 6
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    def evaluate():
        head.eval()
        with torch.no_grad(): z, _ = head(H0[va]); m = retrieval_metrics(z, img[va])
        head.train(); return m

    print("before training:", {k: round(v, 3) for k, v in evaluate().items()})
    pbar = tqdm(range(a.epochs), desc="training")
    for ep_i in pbar:
        perm = tr[torch.randperm(len(tr))]; losses = []
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs].to(device)
            if len(idx) < 8: continue
            H = torch.where(torch.rand(len(idx), 1, device=device) < 0.5, H0[idx], H1[idx])
            z, _ = head(H)
            logits = scale.exp().clamp(max=100) * (z @ img[idx].T)
            tgt = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        m = evaluate() if (ep_i % 5 == 4 or ep_i == a.epochs - 1) else None
        pbar.set_postfix(loss=f"{sum(losses)/len(losses):.3f}", **({"val_top1": f"{m['top1']:.3f}", "val_top5": f"{m['top5']:.3f}"} if m else {}))
        if m: tqdm.write(f"  epoch {ep_i+1}: loss={sum(losses)/len(losses):.3f} val {({k: round(v,3) for k,v in m.items()})}")

    banner("5d: results")
    m = evaluate(); print("final val retrieval:", {k: round(v, 3) for k, v in m.items()})
    print(f"chance: top1={1/n_val:.3f} top5={5/n_val:.3f}  -> EXPECT clearly above chance; if equal to chance something is wrong")
    head.eval()
    with torch.no_grad(): z, _ = head(H0[va]); sim = z @ img[va].T
    for qi in range(3):
        top = sim[qi].topk(3).indices.tolist()
        print(f"\nquery caption: {meta[int(va[qi])]['captions'][0]!r}")
        for r_i, j in enumerate(top): print(f"   #{r_i+1} retrieved image's caption: {meta[int(va[j])]['captions'][0]!r}" + ("   <-- correct" if j == qi else ""))
    torch.save({"head": head.state_dict(), "d_model": d}, f"{CKPT}/head_step5.pt")
    print(f"\nsaved {CKPT}/head_step5.pt\nSTEP 5 OK")

if __name__ == "__main__":
    main()
