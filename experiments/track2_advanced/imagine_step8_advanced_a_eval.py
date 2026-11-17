# Eval-only (no training, a few minutes): where does the trained Phase-A model have HEADROOM?
# Compares true image / wrong image / blank tokens per family on any split, e.g. the OOD split (bigger grids, longer walks, bigger numbers).
# If blank-token accuracy is ~100% on a family, imagination cannot help there; look for families where it is clearly lower.
# Run:  python imagine_step8_advanced_a_eval.py --data spatial2 --vis grid --grid 3 --split test_ood --n 2000
import argparse, os, torch
from imagine_adv_lib import *
if globals().get("LIB_VERSION", 0) < 3: raise SystemExit("imagine_adv_lib.py in this folder is outdated (missing suffix_of / track filter). Copy the newest imagine_adv_lib.py next to this script and retry.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--vis", default="global", choices=["global", "grid"]); ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--data", default="spatial2", choices=["scienceqa", "spatial", "spatial2"]); ap.add_argument("--split", default="test"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--track", default="all", choices=["all", "imagine", "see"])
    a = ap.parse_args(); set_seed(0); src = f"{CKPT}/adv_a_{tag_of(a)}{suffix_of(a)}"
    if not os.path.exists(f"{src}/reader.pt"): raise SystemExit(f"no Phase A checkpoint at {src}")
    items, vis = prepare_split(a.split, a.n, a.vis, a.grid, source=a.data, track=a.track)
    tok, model = load_lm(a.model); d_model = model.config.hidden_size; ck = torch.load(f"{src}/reader.pt")
    reader = VisualReader(ck["d_v"], ck["n_tok"], d_model, ck["emb_rms"]).to(dev); reader.load_state_dict(ck["reader"])
    pm = load_lora(model, f"{src}/lora"); parts = lm_parts(pm); pm.eval(); reader.eval(); ans_ids, letter_ids = answer_tokens(tok); tokenize_items(tok, items)
    off = len(items) // 2
    conds = {"oracle": lambda idx: reader(vis.get(idx).to(dev)), "wrong": lambda idx: reader(vis.get([(j + off) % len(items) for j in idx]).to(dev)),
             "const": lambda idx: reader(vis.get(idx).to(dev), torch.ones(len(idx), dtype=torch.bool, device=dev))}
    res = eval_conditions(parts, items, ans_ids, letter_ids, conds, desc=a.split); summarize(res, items, f"{a.split} (Phase-A model)")
    contrast(res, "image content is usable   (oracle vs wrong)", "oracle", "wrong"); contrast(res, "image helps vs blank      (oracle vs const)", "oracle", "const")
    breakdown(res, items, "track"); breakdown(res, items, "family")
    print("\nHEADROOM = families where 'const' (text only) is clearly below 100% -> candidates for imagination experiments.\nSTEP 8-ADVANCED-A-EVAL OK")

if __name__ == "__main__":
    main()
