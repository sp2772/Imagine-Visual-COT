# Leak-free spatial dataset v2 for the imagination-head research. Pure PIL, no GPU. Self-audits at the end.
# Design rules (each one fixes a leak found in v1):
#   * the picture shows the SCENE / STATE, never the verdict: no 'fits' text, no green/red answer colouring, no ring on the answer
#   * yes/no families are balanced BY CONSTRUCTION (label chosen first, instance built to match)
#   * choices are shuffled; distractors never depend on the gold in a fixed pattern (answer index ~ uniform, rank ~ uniform)
#   * 'imagine' track: the TEXT fully specifies the scene  -> a student can imagine it, a picture can be the teaching signal
#   * 'see' track:     the text says nothing about the scene -> the image is NECESSARY (blank-image baseline must be ~chance)
#   * grid_walk is multi-step state tracking: every reasoning step has its own state image (--step_images) = per-step imagination targets
#   * test_ood.json: harder than train (bigger grids, longer walks, bigger numbers) to test generalisation
# Run:  python imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000 [--step_images]
import argparse, json, os, random
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
try: from imagine_data import CACHE, banner
except Exception: CACHE = "imagine_cache"; banner = print

ROOT = f"{CACHE}/spatial_cot_v2"
PAL = [(200, 60, 60), (50, 90, 190), (40, 150, 80), (200, 140, 30), (140, 70, 180), (30, 140, 160)]
NAMES = ["red", "blue", "green", "orange", "purple", "teal"]; SHAPES = ["circle", "square", "triangle"]
INK = (30, 30, 30); NEUTRAL_A = (40, 80, 160); NEUTRAL_B = (210, 120, 30)       # fixed colours: never encode the answer

def font(size):
    for p in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()
def canvas(s=256): im = Image.new("RGB", (s, s), (255, 255, 255)); return im, ImageDraw.Draw(im)
def save(im, folder, name):
    os.makedirs(folder, exist_ok=True); p = os.path.join(folder, name); im.save(p, quality=90); return p.replace("\\", "/")
def label(d, xy, text, size=13, fill=INK): d.text(xy, text, fill=fill, font=font(size))
def shape(d, kind, c, r, fill):
    x, y = c
    if kind == "circle": d.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=INK, width=2)
    elif kind == "square": d.rectangle((x - r, y - r, x + r, y + r), fill=fill, outline=INK, width=2)
    else: d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=fill, outline=INK)
def choices_with(rng, gold, pool, k):
    others = [x for x in pool if x != gold]; rng.shuffle(others); ch = [gold] + others[:k - 1]; rng.shuffle(ch); return ch, ch.index(gold)
def base(uid, fam, track, q, ch, ans, sol, img, steps, spec):
    return {"id": uid, "family": fam, "track": track, "gate": 1.0, "question": q, "choices": ch, "answer": ans, "hint": "", "solution": sol,
            "has_image": True, "image_path": img, "steps": steps, "spec": spec}

# ------------------------------------------------------------------------------------------------ IMAGINE track
def fit_ball_box(rng, D, uid, ood=False, step_images=False, **_):
    lo, hi = (20, 40) if ood else (6, 18); w, h = rng.randint(lo, hi), rng.randint(lo, hi); m = min(w, h); want = rng.random() < 0.5; gap = 4 if not ood else 8
    d = rng.randint(max(3, m - gap), m) if want else m + rng.randint(1, gap); fits = d <= m
    s = 110 / max(w, h, d); bw, bh = w * s, h * s; x0, y0 = 16, 200 - bh
    def sk(stage):
        im, dr = canvas(); dr.rectangle((x0, y0, x0 + bw, y0 + bh), outline=NEUTRAL_A, width=3)
        label(dr, (x0, 204), f"{w} cm"); label(dr, (x0 + bw + 4, y0 + bh / 2 - 7), f"{h}")
        if stage >= 1:
            r = d * s / 2; cx, cy = (x0 + bw + 30 + r, y0 + bh - r) if stage == 1 else (x0 + bw / 2, y0 + bh / 2)
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), outline=NEUTRAL_B, width=3); label(dr, (cx - 16, cy - r - 16), f"d={d}")
        return im
    ps = [save(sk(i), D, f"{uid}_s{i}.jpg") if (i == 2 or step_images) else None for i in range(3)]
    p = ps[2]; txt = [f"Box: width {w} cm, height {h} cm.", f"The ball needs {d} cm in every direction.", f"Compare {d} with the smaller side {m}: {'fits' if fits else 'does not fit'}."]
    return base(uid, "fit_ball_box", "imagine", f"Can a ball of diameter {d} cm fit inside a box that is {w} cm wide and {h} cm tall?", ["Yes", "No"], 0 if fits else 1,
                " ".join(txt), p, [{"text": t, "image_path": ps[i] or p} for i, t in enumerate(txt)], {"w": w, "h": h, "d": d})

def fit_rect(rng, D, uid, ood=False, **_):
    lo, hi = (20, 40) if ood else (8, 20); W, H = rng.randint(lo, hi), rng.randint(lo, hi); want = rng.random() < 0.5; sm = 3 if not ood else 8
    if want: w, h = rng.randint(sm, W), rng.randint(sm, H)
    elif rng.random() < 0.5: w, h = W + rng.randint(1, 4 if not ood else 8), rng.randint(sm, H + 2)
    else: h, w = H + rng.randint(1, 4 if not ood else 8), rng.randint(sm, W + 2)
    fits = w <= W and h <= H; s = 160 / max(W, H, w, h); im, dr = canvas(); x0, y0 = 40, 40
    dr.rectangle((x0, y0, x0 + W * s, y0 + H * s), outline=NEUTRAL_A, width=3); dr.rectangle((x0, y0, x0 + w * s, y0 + h * s), outline=NEUTRAL_B, width=2)
    label(dr, (x0, 12), f"outer {W} x {H}", fill=NEUTRAL_A); label(dr, (x0, 232), f"inner {w} x {h}", fill=NEUTRAL_B)
    p = save(im, D, f"{uid}.jpg"); t = f"Need {w} <= {W} and {h} <= {H}: {'yes' if fits else 'no'}."
    return base(uid, "fit_rect", "imagine", f"Can a {w} by {h} cm rectangle be placed inside a {W} by {H} cm rectangle without rotating it?", ["Yes", "No"], 0 if fits else 1, t, p, [{"text": t, "image_path": p}], {"W": W, "H": H, "w": w, "h": h})

def triangle(rng, D, uid, ood=False, **_):
    lo, hi = (16, 40) if ood else (3, 15); want = rng.random() < 0.5
    for _t in range(200):
        a, b, c = (rng.randint(lo, hi) for _ in range(3)); ok = a + b > c and a + c > b and b + c > a
        if ok == want: break
    L, p_, q_ = sorted([a, b, c], reverse=True); s = 190 / L; im, dr = canvas(); y = 215
    dr.line((30, y, 30 + L * s, y), fill=INK, width=3)
    dr.arc((30 - p_ * s, y - p_ * s, 30 + p_ * s, y + p_ * s), 180, 360, fill=NEUTRAL_A, width=3); dr.arc((30 + L * s - q_ * s, y - q_ * s, 30 + L * s + q_ * s, y + q_ * s), 180, 360, fill=NEUTRAL_B, width=3)
    label(dr, (30, y + 6), f"longest = {L}"); pth = save(im, D, f"{uid}.jpg"); t = f"Two shorter sticks {p_} and {q_} must together exceed the longest {L}: {p_ + q_} vs {L}."
    sides = [a, b, c]
    return base(uid, "triangle", "imagine", f"Can three sticks of length {sides[0]} cm, {sides[1]} cm and {sides[2]} cm form a triangle?", ["Yes", "No"], 0 if ok else 1, t, pth, [{"text": t, "image_path": pth}], {"sides": sides})

def draw_cells(g, objs, s=256):
    im, dr = canvas(s); cell = 66 if g == 3 else 50; ox = oy = 20
    for i in range(g + 1): dr.line((ox, oy + i * cell, ox + g * cell, oy + i * cell), fill=(200, 200, 200)); dr.line((ox + i * cell, oy, ox + i * cell, oy + g * cell), fill=(200, 200, 200))
    for o in objs: shape(dr, o["shape"], (ox + o["c"] * cell + cell // 2, oy + o["r"] * cell + cell // 2), cell // 4, PAL[NAMES.index(o["color"])])
    return im

def sample_objs(rng, g, n):
    cells = [(r, c) for r in range(g) for c in range(g)]; rng.shuffle(cells); combos = [(c, s) for c in NAMES for s in SHAPES]; rng.shuffle(combos)
    return [{"r": cells[i][0], "c": cells[i][1], "color": combos[i][0], "shape": combos[i][1]} for i in range(n)]

def spatial_grid(rng, D, uid, ood=False, **_):
    g, n = (4, rng.randint(5, 7)) if ood else (3, rng.randint(3, 5))
    for _t in range(200):
        objs = sample_objs(rng, g, n); mode = rng.choice(["leftmost", "topmost", "color_at"])
        cols = Counter(o["c"] for o in objs); rows = Counter(o["r"] for o in objs)
        if mode == "leftmost" and cols[min(cols)] == 1 or mode == "topmost" and rows[min(rows)] == 1 or mode == "color_at": break
    nm = lambda o: f"{o['color']} {o['shape']}"; lines = "\n".join(f"- the {nm(o)} is in row {o['r']+1}, column {o['c']+1}" for o in objs)
    head = f"A {g}x{g} grid (row 1 is the top, column 1 is the left) contains these objects:\n{lines}\n"
    names = [nm(o) for o in objs]; extras = [f"{c} {s_}" for c in NAMES for s_ in SHAPES if f"{c} {s_}" not in names]
    if mode in ("leftmost", "topmost"):
        gold_o = min(objs, key=(lambda o: o["c"]) if mode == "leftmost" else (lambda o: o["r"])); gold = nm(gold_o)
        q = head + f"Which object is the {mode}?"; ch, ans = choices_with(rng, gold, names + extras, 4)
    else:
        t = rng.choice(objs); gold = t["color"]; q = head + f"What color is the object in row {t['r']+1}, column {t['c']+1}?"; ch, ans = choices_with(rng, gold, NAMES, 4)
    p = save(draw_cells(g, objs), D, f"{uid}.jpg")
    return base(uid, "spatial_grid", "imagine", q, ch, ans, gold, p, [{"text": "Sketch the grid from the description.", "image_path": p}], {"objs": objs, "mode": mode})

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
def draw_board(N, walls, agent, trail=()):
    im, dr = canvas(); cell = 200 // N; ox = oy = 36
    for (r, c) in walls: dr.rectangle((ox + (c - 1) * cell, oy + (r - 1) * cell, ox + c * cell, oy + r * cell), fill=(95, 95, 95))
    for i in range(N + 1): dr.line((ox, oy + i * cell, ox + N * cell, oy + i * cell), fill=(190, 190, 190)); dr.line((ox + i * cell, oy, ox + i * cell, oy + N * cell), fill=(190, 190, 190))
    for i in range(N): label(dr, (ox + i * cell + cell // 2 - 4, 14), str(i + 1), 11); label(dr, (14, oy + i * cell + cell // 2 - 6), str(i + 1), 11)
    ctr = lambda p: (ox + (p[1] - 1) * cell + cell // 2, oy + (p[0] - 1) * cell + cell // 2)
    if len(trail) > 1: dr.line([ctr(p) for p in trail], fill=(150, 170, 220), width=3)
    x, y = ctr(agent); r = max(5, cell // 3); dr.ellipse((x - r, y - r, x + r, y + r), fill=(40, 90, 190), outline=INK, width=2); return im

def grid_walk(rng, D, uid, ood=False, step_images=False, **_):
    N = rng.choice([7, 8]) if ood else rng.choice([4, 5, 6]); T = rng.randint(8, 12) if ood else rng.randint(2, 6)
    cells = [(r, c) for r in range(1, N + 1) for c in range(1, N + 1)]; start = rng.choice(cells); walls = set(rng.sample([c for c in cells if c != start], rng.randint(N // 2, N + 2)))
    moves = [rng.choice(list(DIRS)) for _ in range(T)]; pos = start; path = [pos]; log = []
    for mv in moves:
        r, c = pos[0] + DIRS[mv][0], pos[1] + DIRS[mv][1]; blocked = not (1 <= r <= N and 1 <= c <= N) or (r, c) in walls
        if not blocked: pos = (r, c)
        path.append(pos); log.append(f"{mv} -> ({pos[0]},{pos[1]})" + (" (blocked)" if blocked else ""))
    free = [c for c in cells if c not in walls]; gold = f"({pos[0]},{pos[1]})"; ch, ans = choices_with(rng, gold, [f"({a},{b})" for a, b in free], 4)
    wt = ", ".join(f"({a},{b})" for a, b in sorted(walls))
    q = (f"A {N}x{N} grid (row 1 is the top, column 1 is the left). Walls are at {wt}. An agent starts at ({start[0]},{start[1]}) and tries these moves in order: {', '.join(moves)}. "
         f"A move into a wall or off the grid is ignored. Where does the agent end up?")
    scene = save(draw_board(N, walls, start), D, f"{uid}_scene.jpg"); steps = [{"text": f"Start at ({start[0]},{start[1]}).", "image_path": scene}]
    for i, t in enumerate(log):
        steps.append({"text": f"Move {i+1}: {t}.", "image_path": save(draw_board(N, walls, path[i + 1], path[:i + 2]), D, f"{uid}_m{i+1}.jpg") if step_images else None})
    return base(uid, "grid_walk", "imagine", q, ch, ans, " ".join(s["text"] for s in steps), scene, steps, {"N": N, "T": T, "walls": sorted(walls), "start": start, "moves": moves, "path": path})

# ------------------------------------------------------------------------------------------------ SEE track (image necessary)
def scatter(rng, n, min_d=36):
    while True:
        pts = []
        for _ in range(n * 80):
            p = (rng.randint(24, 232), rng.randint(24, 232))
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_d ** 2 for q in pts): pts.append(p)
            if len(pts) == n: return pts
        min_d = max(18, min_d - 4)

def count_see(rng, D, uid, **_):
    gold = rng.randint(0, 8); ac, ash = rng.choice(NAMES), rng.choice(SHAPES); objs = [(ac, ash)] * gold; extra = rng.randint(3, 8)
    while len(objs) < gold + extra:
        o = (rng.choice(NAMES), rng.choice(SHAPES))
        if o != (ac, ash): objs.append(o)
    pts = scatter(rng, len(objs)); im, dr = canvas(); order = list(range(len(pts))); rng.shuffle(order)
    for (c, s_), p in zip([objs[i] for i in order], pts): shape(dr, s_, p, 13, PAL[NAMES.index(c)])
    ch, ans = choices_with(rng, str(gold), [str(x) for x in range(9)], 4); p = save(im, D, f"{uid}.jpg")
    return base(uid, "count_see", "see", f"How many {ac} {ash}s are in the figure?", ch, ans, str(gold), p, [{"text": "Count them in the figure.", "image_path": p}], {"gold": gold})

def mark_see(rng, D, uid, **_):
    objs = sample_objs(rng, 3, rng.randint(3, 5)); t = rng.choice(objs); im = draw_cells(3, objs); dr = ImageDraw.Draw(im); cell = 66
    cx, cy = 20 + t["c"] * cell + cell // 2, 20 + t["r"] * cell + cell // 2; dr.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=(0, 0, 0), width=3)
    if rng.random() < 0.5: q = "What color is the object inside the black ring?"; ch, ans = choices_with(rng, t["color"], NAMES, 4)
    else: q = "What shape is the object inside the black ring?"; ch, ans = choices_with(rng, t["shape"], SHAPES, 3)
    p = save(im, D, f"{uid}.jpg"); return base(uid, "mark_see", "see", q, ch, ans, ch[ans], p, [{"text": "Look at the ringed object.", "image_path": p}], {"target": t})

def rel_see(rng, D, uid, **_):
    a, b = rng.sample([(c, s) for c in NAMES for s in SHAPES], 2); vertical = rng.random() < 0.5; want = rng.random() < 0.5
    pts = scatter(rng, rng.randint(2, 4), 40); pa, pb = pts[0], pts[1]; key = 1 if vertical else 0
    if abs(pa[key] - pb[key]) < 25: pb = (pb[0] + (30 if not vertical else 0), pb[1] + (30 if vertical else 0))
    if (pa[key] < pb[key]) != want: pa, pb = pb, pa
    im, dr = canvas(); shape(dr, a[1], pa, 13, PAL[NAMES.index(a[0])]); shape(dr, b[1], pb, 13, PAL[NAMES.index(b[0])])
    for p, (c, s_) in zip(pts[2:], [(rng.choice(NAMES), rng.choice(SHAPES)) for _ in pts[2:]]):
        if (c, s_) not in (a, b): shape(dr, s_, p, 13, PAL[NAMES.index(c)])
    rel = "above" if vertical else "to the left of"; p = save(im, D, f"{uid}.jpg")
    return base(uid, "rel_see", "see", f"Is the {a[0]} {a[1]} {rel} the {b[0]} {b[1]}?", ["Yes", "No"], 0 if want else 1, "yes" if want else "no", p, [{"text": "Compare positions in the figure.", "image_path": p}], {})

FAMS = [(fit_ball_box, .15), (fit_rect, .10), (triangle, .10), (spatial_grid, .15), (grid_walk, .20), (count_see, .12), (mark_see, .08), (rel_see, .10)]
OOD_FAMS = [(fit_ball_box, .25), (fit_rect, .2), (triangle, .15), (spatial_grid, .15), (grid_walk, .25)]

def pick(rng, fams):
    u, acc = rng.random() * sum(w for _, w in fams), 0.0
    for f, w in fams:
        acc += w
        if u <= acc: return f
    return fams[-1][0]

def dump(name, n, seed, fams, ood=False, step_images=False):
    rng = random.Random(seed); D = os.path.join(ROOT, "images", name); os.makedirs(D, exist_ok=True); items = []
    for i in range(n):
        it = pick(rng, fams)(rng, D, f"{name}_{i:06d}", ood=ood, step_images=step_images); it["split"] = name; items.append(it)
        if (i + 1) % 2000 == 0: print(f"  {name}: {i+1}/{n}", flush=True)
    json.dump(items, open(os.path.join(ROOT, f"{name}.json"), "w", encoding="utf-8"))
    print(f"[{name}] {n} items | families {dict(Counter(x['family'] for x in items))}", flush=True); return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=40000); ap.add_argument("--n_val", type=int, default=3000); ap.add_argument("--n_test", type=int, default=3000); ap.add_argument("--n_ood", type=int, default=2000)
    ap.add_argument("--step_images", action="store_true", help="also render one state image per walk step (more disk; needed for multi-step imagination)")
    a = ap.parse_args(); banner("building spatial_cot_v2 (leak-free)"); os.makedirs(ROOT, exist_ok=True)
    dump("train", a.n_train, 10, FAMS, step_images=a.step_images); val = dump("val", a.n_val, 11, FAMS, step_images=a.step_images); dump("test", a.n_test, 12, FAMS, step_images=a.step_images)
    if a.n_ood: dump("test_ood", a.n_ood, 13, OOD_FAMS, ood=True, step_images=a.step_images)
    json.dump(vars(a), open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
    try:
        from imagine_audit_data import audit; audit(val, "SELF-AUDIT of val.json")
    except Exception as e: print("self-audit skipped:", e)
    print(f"\nsaved under {ROOT}\nBUILD V2 OK")

if __name__ == "__main__":
    main()
