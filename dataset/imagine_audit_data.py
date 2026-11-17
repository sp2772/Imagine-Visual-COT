# Leak audit for any dataset JSON in the {question, choices, answer, family, image_path} schema. No GPU, no torch.
# It asks: can the answer be guessed WITHOUT doing the task?  Four cheap attackers per family:
#   1) answer-index bias        2) rank-of-answer bias for numeric choices (e.g. 'gold = 2nd smallest')
#   3) question/choice-set lookup table (memorised templates)         4) one cheap IMAGE feature (colour counts, ink) on yes/no families
# A family is flagged if an attacker beats chance by a clear margin. A good dataset has NO flags in the 'see' track
# and no image flags anywhere.   Usage:  python imagine_audit_data.py imagine_cache/spatial_cot_v2/val.json [more.json ...]
import json, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

FEAT_NAMES = ["mean-red", "mean-green", "mean-blue", "ink", "GREEN-pixels", "RED-pixels", "blue-pixels", "dark-pixels", "orange-pixels"]

def feats(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.int16); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ink = (a.sum(-1) < 600)
    return np.array([r.mean(), g.mean(), b.mean(), ink.mean(), ((g > r + 40) & (g > b + 20)).mean(), ((r > g + 60) & (r > b + 60)).mean(),
                     ((b > r + 40) & (b > g + 20)).mean(), (a.sum(-1) < 150).mean(), ((r > 150) & (g > 120) & (b < 90)).mean()])

def stump(X, y, Xt, yt):
    best = (0.5, None)
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 19))):
            for pol in (1, -1):
                acc = ((pol * (X[:, j] - t) > 0).astype(int) == y).mean()
                if acc > best[0]: best = (acc, (j, t, pol))
    if best[1] is None: return 0.5, "-"
    j, t, pol = best[1]; return ((pol * (Xt[:, j] - t) > 0).astype(int) == yt).mean(), FEAT_NAMES[j]

def audit(items, name, margin=0.10):
    print(f"\n{'='*72}\n{name}  (n={len(items)})\n{'='*72}"); flags = 0; by = defaultdict(list)
    for it in items: by[it.get("family", "?")].append(it)
    for fam, its in sorted(by.items()):
        n = len(its); k = sum(len(i["choices"]) for i in its) / n; chance = 1 / k; half = n // 2; msgs, notes = [], []
        idx = Counter(i["answer"] for i in its); top, share = idx.most_common(1)[0]
        if share / n > chance + margin: msgs.append(f"answer-INDEX bias: index {top} is correct {share/n:.0%} (chance {chance:.0%})")
        # numeric rank bias
        ranks = Counter()
        for i in its:
            try: v = [float(c.split()[0]) for c in i["choices"]]
            except Exception: ranks = None; break
            ranks[sorted(v).index(v[i["answer"]]) if len(set(v)) == len(v) else -1] += 1
        if ranks:
            r_top, r_n = ranks.most_common(1)[0]
            if r_n / n > chance + margin: msgs.append(f"numeric RANK bias: gold is the #{r_top+1}-smallest option {r_n/n:.0%} (chance {chance:.0%})")
        # lookup tables (train on first half, test on second half)
        for nm, keyf in (("question", lambda i: i["question"]), ("choice-set", lambda i: tuple(sorted(i["choices"])))):
            tab = defaultdict(Counter)
            for i in its[:half]: tab[keyf(i)][i["choices"][i["answer"]]] += 1
            hit = tot = 0
            for i in its[half:]:
                kf = keyf(i)
                if kf in tab: tot += 1; hit += tab[kf].most_common(1)[0][0] == i["choices"][i["answer"]]
            if tot >= 30 and hit / tot > chance + 0.2 and tot / (n - half) > 0.3: msgs.append(f"{nm} lookup solves {hit/tot:.0%} of {tot} seen items (chance {chance:.0%})")
        # image stump on yes/no families
        if all(len(i["choices"]) == 2 for i in its) and n >= 200 and "image_path" in its[0]:
            try:
                X = np.stack([feats(i["image_path"]) for i in its[:600]]); y = np.array([i["answer"] for i in its[:600]]); h = len(y) // 2
                acc, fn = stump(X[:h], y[:h], X[h:], y[h:])
                if acc > 0.62 and fn in ("GREEN-pixels", "RED-pixels"): msgs.append(f"IMAGE alone predicts the answer {acc:.0%} from {fn}: answer-coded colour (e.g. green=fits / red=no)  <-- LEAK")
                elif acc > 0.62: notes.append(f"image alone predicts {acc:.0%} from '{fn}': the sketch encodes the geometry to scale (expected for 'imagine' sketches; check it is geometry, not a verdict label)")
            except Exception as e: msgs.append(f"(image test skipped: {e})")
        status = "OK   " if not msgs else "LEAK?"; flags += bool(msgs)
        print(f"[{status}] {fam:18s} n={n:5d} choices~{k:.1f} track={its[0].get('track','?')}")
        for m in msgs: print("          -", m)
        for m in notes: print("          . note:", m)
    print(f"\n=> {flags} of {len(by)} families flagged" + ("  (fix before training!)" if flags else "  (clean under these attackers; still eyeball some images)"))
    return flags

if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    total = 0
    for p in sys.argv[1:]: total += audit(json.load(open(p, encoding="utf-8")), p)
    print("AUDIT OK" if total == 0 else f"AUDIT: {total} flagged family(ies)")
