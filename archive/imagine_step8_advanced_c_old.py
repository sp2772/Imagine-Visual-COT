# Step 8-advanced-C: FINAL held-out evaluation on the official ScienceQA TEST split (never used for training or model selection)
#   1) full condition table with paired bootstrap CIs   2) per-subject accuracy   3) a gallery of what the student "imagines"
# Run:  python imagine_step8_advanced_c.py --vis grid --grid 3 --n_test 800     (same --model/--vis/--grid as A and B)
import argparse, os, textwrap, torch, torch.nn.functional as F
from imagine_adv_lib import *

def centred(x): return F.normalize(x - x.mean(0, keepdim=True), dim=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--vis", default="global", choices=["global", "grid"]); ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=800); ap.add_argument("--n_show", type=int, default=6)
    a = ap.parse_args(); set_seed(0); tag = tag_of(a); srcA = f"{CKPT}/adv_a_{tag}"; srcB = f"{CKPT}/adv_b_{tag}"
    for p in (f"{srcA}/reader.pt", f"{srcB}/head.pt"): 
        if not os.path.exists(p): raise SystemExit(f"missing {p}: run Phase A and Phase B with the same --model/--vis/--grid first.")
    banner("C1: load test split + trained components")
    te, vte = prepare_split("test", a.n_test, a.vis, a.grid); tok, model = load_lm(a.model); d_model = model.config.hidden_size
    ckA = torch.load(f"{srcA}/reader.pt"); reader = VisualReader(ckA["d_v"], ckA["n_tok"], d_model, ckA["emb_rms"]).to(dev); reader.load_state_dict(ckA["reader"])
    ckB = torch.load(f"{srcB}/head.pt"); head = VisualImaginationHead(d_model, ckB["d_v"], ckB["n_tok"]).to(dev); head.load_state_dict(ckB["head"])
    pm = load_lora(model, f"{srcA}/lora"); parts = lm_parts(pm); pm.eval(); reader.eval(); head.eval()
    ans_ids, letter_ids = answer_tokens(tok); tokenize_items(tok, te); H = prompt_hidden(parts, te, desc="hidden (test)")
    print(f"test questions: {len(te)} | trained for {ckA['epoch']} reader epochs and {ckB['epoch']} imagination epochs")

    off = len(te) // 2
    imagine = lambda idx: reader(head(H[torch.as_tensor(idx)].to(dev)))
    conds = {"oracle": lambda idx: reader(vte.get(idx).to(dev)), "own": imagine,
             "own_shuffled": lambda idx: imagine([(j + off) % len(te) for j in idx]),
             "wrong": lambda idx: reader(vte.get([(j + off) % len(te) for j in idx]).to(dev)),
             "const": lambda idx: reader(vte.get(idx).to(dev), torch.ones(len(idx), dtype=torch.bool, device=dev))}
    banner("C2: held-out TEST results")
    with torch.no_grad(): res = eval_conditions(parts, te, ans_ids, letter_ids, conds, desc="test")
    summarize(res, te, "TEST split (blank-slate = 'own': question text only, no image)")
    print("\npaired bootstrap (95% CI; positive = first-named condition is better):")
    contrast(res, "own imagination vs blank        (const - own)", "own", "const")
    contrast(res, "own is example-specific  (own_shuffled - own)", "own", "own_shuffled")
    contrast(res, "own vs a random real image      (wrong - own)", "own", "wrong")
    contrast(res, "headroom: true image vs own     (own - oracle)", "oracle", "own")
    contrast(res, "[reference] image content usable (wrong - oracle)", "oracle", "wrong")
    contrast(res, "[reference] true image vs blank (const - oracle)", "oracle", "const")

    banner("C3: accuracy by subject")
    for subj in sorted({i["subject"] for i in te}):
        ix = [j for j, i in enumerate(te) if i["subject"] == subj]
        print(f"  {subj or '(none)':18s} n={len(ix):4d} | " + " | ".join(f"{k}={sum(res[k]['ok'][j] for j in ix)/len(ix)*100:5.1f}%" for k in ("const", "own", "oracle")))

    banner("C4: what does the student imagine?  (retrieval of real test images from the imagined tokens)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    with torch.no_grad():
        imag = head(H.to(dev)).float().mean(1); true = vte.target(list(range(len(te)))).to(dev).mean(1); mu = true.mean(0, keepdim=True)
        Zn = F.normalize(imag - mu, dim=-1); Tn = F.normalize(true - mu, dim=-1); sim = Zn @ Tn.T
    top1 = (sim.argmax(1) == torch.arange(len(te), device=dev)).float().mean().item(); top5 = (sim.topk(5, 1).indices == torch.arange(len(te), device=dev)[:, None]).any(1).float().mean().item()
    print(f"imagined tokens -> true image retrieval over {len(te)} test images: top1={top1:.3f} top5={top5:.3f} (chance {1/len(te):.3f} / {5/len(te):.3f})")
    fig, axes = plt.subplots(a.n_show, 5, figsize=(16, 3.1 * a.n_show), squeeze=False)
    for r in range(a.n_show):
        axes[r, 0].axis("off"); axes[r, 0].text(0, 0.5, textwrap.fill(te[r]["question"], 34) + f"\n\n[own p(correct)] see table\nanswer: {te[r]['choices'][te[r]['answer']][:30]}", fontsize=8, va="center")
        axes[r, 1].imshow(Image.open(te[r]["image_path"])); axes[r, 1].set_title("TRUE image", fontsize=8); axes[r, 1].axis("off")
        for c, j in enumerate(sim[r].topk(3).indices.tolist()):
            axes[r, c + 2].imshow(Image.open(te[j]["image_path"])); axes[r, c + 2].axis("off")
            axes[r, c + 2].set_title(f"imagined #{c+1}: " + textwrap.fill(te[j]["question"][:44], 26), fontsize=7)
    plt.tight_layout(); fn = f"{OUT}/adv_c_{tag}_imagination.png"; plt.savefig(fn, dpi=100); print("saved gallery ->", fn)
    print("\nSTEP 8-ADVANCED-C OK")

if __name__ == "__main__":
    main()
