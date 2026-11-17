# Step 7: "image decoder" stand-in = nearest-neighbour retrieval over the COCO gallery using z_t.
# Lets you SEE what the head imagines. Gate decides show/hide.
# Run: python imagine_step7.py     (needs steps 5 and 6)
import argparse, os, textwrap, torch
from PIL import Image
from imagine_step1 import load_backbone, batch_hidden, device
from imagine_step2 import ImaginationHead
from imagine_data import *

PROBES = ["A ball sitting next to a box on a table.", "A red double decker bus driving on a city street.",
          "A dog running on a sandy beach.", "A plate of food with broccoli and rice.",
          "The capital of France is Paris."]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--topk", type=int, default=3)
    a = ap.parse_args()
    banner("STEP 7: imagination -> retrieved image")
    ck = f"{CKPT}/head_step6.pt" if os.path.exists(f"{CKPT}/head_step6.pt") else f"{CKPT}/head_step5.pt"
    print("loading head:", ck)
    gal = torch.load(f"{CACHE}/coco_clip_embs.pt"); G = gal["embs"].to(device); paths, caps = gal["paths"], gal["captions"]
    print("gallery:", tuple(G.shape))
    tok, model = load_backbone()
    H = batch_hidden(tok, model, [p + "\n" for p in PROBES], desc="probe hidden").to(device)
    head = ImaginationHead(model.config.hidden_size).to(device); head.load_state_dict(torch.load(ck, map_location=device)["head"]); head.eval()
    with torch.no_grad(): z, g = head(H)
    top = (z @ G.T).topk(a.topk, dim=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = len(PROBES); fig, axes = plt.subplots(R, a.topk + 1, figsize=(3.2 * (a.topk + 1), 3.2 * R), squeeze=False)
    for r in range(R):
        show = g[r].item() > a.thr
        verdict = "SHOW image" if show else "hidden (latent only; shown for debug)"
        print(f"\n[{r}] {PROBES[r]!r}\n    gate={g[r].item():.3f} -> {verdict}")
        axes[r, 0].axis("off"); axes[r, 0].text(0, 0.5, textwrap.fill(PROBES[r], 22) + f"\n\ngate={g[r].item():.2f}\n{verdict}", fontsize=9, va="center")
        for c in range(a.topk):
            j = int(top.indices[r, c]); axes[r, c + 1].imshow(Image.open(paths[j])); axes[r, c + 1].axis("off")
            axes[r, c + 1].set_title(textwrap.fill(caps[j][0], 30), fontsize=7)
            print(f"      #{c+1} sim={top.values[r,c].item():.3f} caption: {caps[j][0]!r}")
    plt.tight_layout(); out = f"{OUT}/step7_retrieval.png"; plt.savefig(out, dpi=110)
    print(f"\nsaved montage -> {out}  (open it and check the images match the text)")
    print("To get REAL generated images later: feed z_t (512-d CLIP space) into an unCLIP/Stable-Diffusion image-variations decoder.")
    print("STEP 7 OK")

if __name__ == "__main__":
    main()
