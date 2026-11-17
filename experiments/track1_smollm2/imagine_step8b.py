# Step 8b: CONTROL experiments on the step-8 checkpoint (no training, takes ~1 minute).
# Question: does the fed-back imagination carry example-specific information, or does the extra token just act like a soft prompt?
#   feedback : I_t = proj(z_t)                     (the real method)
#   shuffled : I_t = proj(z of ANOTHER example)    (right "kind" of token, wrong content)
#   random_z : I_t = proj(random unit vector)
#   constant : I_t = img_start                     (same token every step)
# Run: python imagine_step8b.py
import torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, load_chain_parts
from imagine_step8 import load_data, amp
from imagine_data import *

@torch.no_grad()
def main():
    set_seed(0); banner("STEP 8b: is the imagination informative?")
    tok, model = load_backbone(); d = model.config.hidden_size
    chain = FeedbackChain(model, d).to(device); load_chain_parts(chain, f"{CKPT}/chain_step8.pt"); chain.eval()
    _, val = load_data(400, 80, tok)                      # same val split as step 8
    runs = [chain(x["segs"], x["masks"]) for x in tqdm(val, desc="feedback")]
    zs = [r["z"] for r in runs]; n = len(val)
    res = {"feedback": [r["lm_loss"].item() for r in runs], "shuffled": [], "random_z": [], "constant": []}
    for i, x in enumerate(tqdm(val, desc="controls")):
        res["shuffled"].append(chain(x["segs"], x["masks"], z_swap=zs[(i + 1) % n])["lm_loss"].item())
        rnd = [F.normalize(torch.randn(zs[0].size(-1), device=device), dim=-1) for _ in x["segs"]]
        res["random_z"].append(chain(x["segs"], x["masks"], z_swap=rnd)["lm_loss"].item())
        res["constant"].append(chain(x["segs"], x["masks"], mode="constant")["lm_loss"].item())
    m = {k: sum(v) / n for k, v in res.items()}
    print("\nmean LM loss on val (lower is better):")
    for k in ("feedback", "shuffled", "random_z", "constant"): print(f"  {k:9s} {m[k]:.4f}")
    print(f"\ninformation gain   (shuffled - feedback): {m['shuffled'] - m['feedback']:+.4f}   <- the part that depends on THIS example's imagination")
    print(f"soft-prompt effect (constant - shuffled): {m['constant'] - m['shuffled']:+.4f}   <- gain from just having a trained extra token")
    gold = torch.cat([F.cosine_similarity(x["step_emb"][1:], x["step_emb"][:-1], dim=-1) for x in val])
    mine = torch.cat([F.cosine_similarity(z[1:1 + x["n_steps"]][1:], z[1:1 + x["n_steps"]][:-1], dim=-1) for z, x in zip(zs, val)])
    print(f"\nconsecutive-step cosine: model z = {mine.mean():.3f} | CLIP embeddings of the gold step texts = {gold.mean():.3f}")
    print("(collapse means z is MORE similar across steps than the gold texts themselves; matching the gold value is healthy)")
    print("STEP 8b OK")

if __name__ == "__main__":
    main()
