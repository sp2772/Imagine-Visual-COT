# Step 8d: is the effect REAL or noise?  (no training, ~10 min)
#  1) fresh held-out image questions (never seen in step 8c), more of them than before
#  2) paired bootstrap 95% confidence intervals for every contrast
#  3) a sharper readout: ANSWER-FIRST NLL = how well the model predicts the answer letter right after the question
#     (no gold reasoning steps in context), which is where image content should matter most.
# Contrasts are written as (worse - better): positive = the second-named condition helps.
# Run: python imagine_step8d.py --n_eval 150
import argparse, json, math, os, torch
from tqdm import tqdm
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, load_chain_parts
from imagine_step8 import prepare, LETTERS
from imagine_data import *

def key(it): return (it["question"], tuple(it["choices"]), it["hint"])

def boot(diff, n=4000, seed=0):
    d = torch.tensor(diff, dtype=torch.float32); g = torch.Generator().manual_seed(seed)
    means = d[torch.randint(0, len(d), (n, len(d)), generator=g)].mean(1)
    return d.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item(), (d > 0).float().mean().item()

def letter_idx(tok, d):
    L = LETTERS[d["it"]["answer"]]
    for j, t in enumerate(d["segs"][-1].tolist()):
        if tok.decode([t]).strip() == L: return j
    return None

@torch.no_grad()
def run_conditions(chain, items, get_segs, get_loss, desc):
    """returns {cond: [loss per item]} for own / own_shuffled / oracle / oracle_wrong / constant."""
    m = len(items); own = []
    for d in tqdm(items, desc=f"{desc}: own", leave=False): own.append(chain(*get_segs(d)))
    zs = [r["z"] for r in own]; res = {"own": [get_loss(r, d) for r, d in zip(own, items)]}
    res["own_shuffled"] = [get_loss(chain(*get_segs(d), z_swap=zs[(i + 1) % m]), d) for i, d in enumerate(tqdm(items, desc=f"{desc}: shuffled", leave=False))]
    res["oracle"] = [get_loss(chain(*get_segs(d), oracle=d["img_emb"]), d) for d in tqdm(items, desc=f"{desc}: oracle", leave=False)]
    res["oracle_wrong"] = [get_loss(chain(*get_segs(d), oracle=items[(i + 1) % m]["img_emb"]), d) for i, d in enumerate(tqdm(items, desc=f"{desc}: wrong image", leave=False))]
    res["constant"] = [get_loss(chain(*get_segs(d), mode="constant"), d) for d in tqdm(items, desc=f"{desc}: constant", leave=False)]
    return res

def report(title, res):
    print(f"\n--- {title} (n={len(res['own'])}; mean loss: " + " | ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in res.items()) + ")")
    pairs = [("own helps vs no imagination      (constant - own)", "constant", "own"),
             ("own is example-specific          (own_shuffled - own)", "own_shuffled", "own"),
             ("LLM can read true image          (oracle_wrong - oracle)", "oracle_wrong", "oracle"),
             ("headroom left for own imagination (own - oracle)", "own", "oracle")]
    for name, worse, better in pairs:
        mean, lo, hi, win = boot([a - b for a, b in zip(res[worse], res[better])])
        flag = "SUPPORTED" if lo > 0 else ("opposite" if hi < 0 else "not distinguishable from 0")
        print(f"  {name}: {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  wins {win*100:.0f}%  -> {flag}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=150); ap.add_argument("--n_train", type=int, default=480); ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args(); set_seed(0); banner("STEP 8d: significance + answer-first readout")
    from peft import PeftModel
    tok, model = load_backbone(); d_model = model.config.hidden_size
    chain = FeedbackChain(model, d_model, k=a.k).to(device)
    chain.model = PeftModel.from_pretrained(chain.model, f"{CKPT}/lora_step8c").to(device)
    load_chain_parts(chain, f"{CKPT}/chain_step8c.pt"); chain.eval(); print("loaded step 8c LoRA + head/proj")

    half = (a.n_train + a.n_val) // 2; used_path = f"{CACHE}/sqa_{half}_{half}_2.json"
    used = {key(x) for x in json.load(open(used_path))} if os.path.exists(used_path) else set()
    print(f"questions used in step 8c (excluded): {len(used)}")
    items = get_scienceqa(n_img=half + a.n_eval + 60, n_noimg=0, min_steps=2)
    fresh = [x for x in items if key(x) not in used][:a.n_eval]
    print(f"fresh held-out image questions: {len(fresh)}")
    data = prepare(fresh, tok)

    report("CoT setting: LM loss over question+steps+answer (teacher-forced gold steps)",
           run_conditions(chain, data, lambda d: (d["segs"], d["masks"]), lambda r, d: r["lm_loss"].item(), "CoT"))

    af = [d for d in data if letter_idx(tok, d) is not None]
    for d in af: d["j"] = letter_idx(tok, d)
    chance = sum(math.log(len(d["it"]["choices"])) for d in af) / len(af)
    res = run_conditions(chain, af, lambda d: ([d["segs"][0], d["segs"][-1]], [torch.zeros(len(d["segs"][0]), dtype=torch.bool), torch.zeros(len(d["segs"][-1]), dtype=torch.bool)]),
                         lambda r, d: -r["lps"][-1][d["j"]].item(), "answer-first")
    report(f"ANSWER-FIRST: NLL of the answer letter right after the question (chance level = {chance:.3f})", res)
    print("\nHOW TO READ: 'SUPPORTED' = the CI excludes 0, so the effect is unlikely to be noise. 'not distinguishable' = need more data/training.")
    print("If answer-first NLL is near chance for every condition, the model hasn't learned to use visuals for ANSWERING yet -> next step: train an answer-first objective.")
    print("STEP 8d OK")

if __name__ == "__main__":
    main()
