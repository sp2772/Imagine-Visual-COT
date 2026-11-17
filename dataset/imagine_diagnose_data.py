# Why ScienceQA failed Phase A (oracle ≈ wrong image). No GPU needed.
# Run:  python imagine_diagnose_data.py
import json, os, re
from collections import Counter
from imagine_data import CACHE, banner

def load_any():
    cands = [
        f"{CACHE}/adv_sqa_validation_600.json",
        f"{CACHE}/adv_sqa_train_4000.json",
        f"{CACHE}/sqa_400_400_0.json",
    ]
    found = []
    for p in cands:
        if os.path.exists(p):
            items = json.load(open(p, encoding="utf-8"))
            found.append((p, items))
            print(f"loaded {len(items):5d}  {p}")
    if not found:
        raise SystemExit("no ScienceQA cache yet. nothing to diagnose.")
    return found

def ans_text(it):
    ch = it.get("choices") or []
    a = int(it.get("answer", 0))
    return str(ch[a]) if 0 <= a < len(ch) else ""

def leak(it):
    """Answer already sitting in the question/hint (case-insensitive whole phrase)."""
    gold = ans_text(it).strip()
    if len(gold) < 3:
        return False
    blob = (it.get("question", "") + "\n" + it.get("hint", "")).lower()
    return gold.lower() in blob

def long_hint(it, n=80):
    return len((it.get("hint") or "").strip()) >= n

def knowledge_looking(it):
    q = (it.get("question") or "").lower()
    keys = ("which of the following", "what is the capital", "according to", "lecture",
            "hypothesis", "independent variable", "photosynthesis", "atom", "definition")
    return any(k in q for k in keys)

def report(path, items):
    banner(path)
    n = len(items)
    img = sum(1 for i in items if i.get("image_path") or i.get("has_image"))
    nh = sum(1 for i in items if (i.get("hint") or "").strip())
    lh = sum(1 for i in items if long_hint(i))
    lk = sum(1 for i in items if leak(i))
    kn = sum(1 for i in items if knowledge_looking(i))
    nch = Counter(len(i.get("choices") or []) for i in items)
    print(f"n={n}  with_image={img} ({100*img/n:.0f}%)  with_hint={nh} ({100*nh/n:.0f}%)")
    print(f"long hint (>=80 chars): {lh} ({100*lh/n:.0f}%)   <-- hint often duplicates the figure")
    print(f"gold answer string appears in question/hint: {lk} ({100*lk/n:.0f}%)")
    print(f"knowledge/meta questions (text-solvable even without the figure): {kn} ({100*kn/n:.0f}%)")
    print("choice-count histogram:", dict(nch))
    print("\n3 random examples (truncated):")
    for it in items[:3]:
        gold = ans_text(it)
        print("-" * 60)
        print("Q:", re.sub(r"\s+", " ", it.get("question", ""))[:180])
        hint = re.sub(r"\s+", " ", it.get("hint") or "")[:220]
        if hint:
            print("HINT:", hint)
        print("CHOICES:", it.get("choices"))
        print("GOLD:", gold, "| leak=", leak(it), "| long_hint=", long_hint(it))

    print("\nWHY Phase A cannot pass on this split:")
    print("  The Phase A test is: true image vs a *different question's image*.")
    print("  If the letter is already in the hint / world knowledge, swapping the figure")
    print("  does not change the answer  ->  oracle ~= wrong  ->  OPPOSITE / not distinguishable.")
    print("  That is a DATASET failure, not proof that imagination is impossible.")
    print("  Your research_plan_new.pdf example (ball vs box) is a different task:")
    print("  the numbers uniquely specify a sketch, and a wrong sketch implies the opposite answer.")

if __name__ == "__main__":
    for p, items in load_any():
        report(p, items)
    print("\nnext:  python imagine_build_spatial_cot.py --n_train 20000 --n_val 2000 --n_test 2000")
    print("then:  python imagine_step8_advanced_a.py --data spatial --vis grid --grid 3 --epochs 3")
    print("DIAGNOSE OK")
