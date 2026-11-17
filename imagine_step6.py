# Step 6: train the GATE (g = sigmoid(W1 h)): visual-type questions -> high, abstract -> low.
# Labels come from ScienceQA: question has an image (1) vs no image (0). This is a proxy for "visual" vs "abstract".
# Run: python imagine_step6.py --n 400 --epochs 60
import argparse, os, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, batch_hidden, device
from imagine_step2 import ImaginationHead
from imagine_data import *

PROBES = [("Can a ball with a 9 cm diameter fit inside a box that is 10 cm wide and 8 cm tall?", "visual"),
          ("Which of these shapes is a triangle? Look at the picture.", "visual"),
          ("Look at the map. Which ocean is highlighted in blue?", "visual"),
          ("What is the capital of France?", "abstract"),
          ("Is the following sentence a run-on? 'I ran he walked we stopped.'", "abstract"),
          ("If all bats are mammals and all mammals are animals, are all bats animals?", "abstract")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="items per class"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args(); set_seed(0)

    banner("6a: download ScienceQA (image vs no-image questions)")
    items = get_scienceqa(n_img=a.n, n_noimg=a.n)
    fmt = lambda q, h: f"Question: {q}\nContext: {h or 'none'}\n"
    texts = [fmt(it["question"], it["hint"][:300]) for it in items]
    y = torch.tensor([float(it["has_image"]) for it in items])
    print(f"items: {len(items)} | with image: {int(y.sum())} | without: {int((1-y).sum())}")
    print("example (visual) :", texts[int(y.argmax())][:150].replace("\n", " | "))
    print("example (abstract):", texts[int((1 - y).argmax())][:150].replace("\n", " | "))

    banner("6b: hidden states")
    tok, model = load_backbone()
    H = batch_hidden(tok, model, texts, pool="last", max_len=160).to(device); y = y.to(device)
    probe_H = batch_hidden(tok, model, [fmt(q, "") for q, _ in PROBES], desc="probe hidden").to(device)
    d = H.size(1); del model

    head = ImaginationHead(d).to(device)
    p5 = f"{CKPT}/head_step5.pt"
    if os.path.exists(p5): head.load_state_dict(torch.load(p5, map_location=device)["head"]); print("loaded W2 from step 5 (kept frozen)")
    else: print("WARNING: step 5 checkpoint not found; run step 5 first for a meaningful W2")
    for n, p in head.named_parameters(): p.requires_grad_(n.startswith("W1"))
    opt = torch.optim.AdamW(head.W1.parameters(), lr=a.lr, weight_decay=1e-3)

    perm = torch.randperm(len(H), device=device); nv = len(H) // 5
    va, tr = perm[:nv], perm[nv:]
    logit = lambda x: head.W1(head.norm(x)).squeeze(-1)
    def acc(idx):
        with torch.no_grad(): p = (torch.sigmoid(logit(H[idx])) > 0.5).float()
        return (p == y[idx]).float().mean().item()
    print(f"train {len(tr)} | val {nv} | majority-class baseline ~ 0.50 | val acc before training: {acc(va):.3f}")

    banner("6c: train gate")
    pbar = tqdm(range(a.epochs), desc="gate training")
    for ep in pbar:
        p = tr[torch.randperm(len(tr), device=device)]; ls = []
        for i in range(0, len(p), a.bs):
            idx = p[i:i + a.bs]
            loss = F.binary_cross_entropy_with_logits(logit(H[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        pbar.set_postfix(loss=f"{sum(ls)/len(ls):.3f}", train_acc=f"{acc(tr):.3f}", val_acc=f"{acc(va):.3f}")

    banner("6d: results")
    print(f"final: train acc {acc(tr):.3f} | val acc {acc(va):.3f}  (EXPECT well above 0.5)")
    with torch.no_grad():
        g = torch.sigmoid(logit(probe_H)).tolist()
    print("\nhand-written probes (gate should be high for 'visual', low for 'abstract'):")
    for (q, kind), gv in zip(PROBES, g): print(f"  g={gv:.3f} [{kind:8s}] {q}")
    print("\nNOTE: 'has image' is only a proxy label; if probes look odd, that's the proxy, not a bug.")
    torch.save({"head": head.state_dict(), "d_model": d}, f"{CKPT}/head_step6.pt")
    print(f"saved {CKPT}/head_step6.pt\nSTEP 6 OK")

if __name__ == "__main__":
    main()
