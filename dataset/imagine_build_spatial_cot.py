# Build the dataset that research_plan_new.pdf actually needs.
# Paper example: "Can a ball of diameter 9 cm fit in a box 10 cm wide and 8 cm tall?"
# Two tracks in one dump:
#   imagine : the question TEXT specifies the scene (so the head can imagine a sketch)
#   see     : the question does NOT specify the scene (so the image is necessary)
# Also writes abstract/gate=0 items (capitals, pure arithmetic) in a separate file — do NOT mix
# those into Phase A or the reader will learn to ignore figures again.
#
# Run:
#   python imagine_build_spatial_cot.py --n_train 20000 --n_val 2000 --n_test 2000
# Bigger later:  --n_train 80000 --n_val 4000 --n_test 4000
import argparse, json, math, os, random
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from imagine_data import CACHE, banner

ROOT = f"{CACHE}/spatial_cot"
PAL = [(200, 60, 60), (50, 90, 190), (40, 150, 80), (200, 140, 30), (140, 70, 180), (30, 140, 160)]
NAMES = ["red", "blue", "green", "orange", "purple", "teal"]
SHAPES = ["circle", "square", "triangle"]


def font(size):
    for p in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def canvas(bg=(255, 255, 255), s=256):
    im = Image.new("RGB", (s, s), bg)
    return im, ImageDraw.Draw(im)


def save(im, folder, name):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    im.save(p, quality=92)
    return p.replace("\\", "/")


def label(d, xy, text, fill=(20, 20, 20), size=14):
    d.text(xy, text, fill=fill, font=font(size))


def draw_circle(d, c, r, fill, outline=(20, 20, 20), w=2):
    x, y = c
    d.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=w)


def draw_rect(d, box, fill, outline=(20, 20, 20), w=3):
    d.rectangle(box, outline=outline, width=w)
    if fill:
        x0, y0, x1, y1 = box
        d.rectangle((x0 + w, y0 + w, x1 - w, y1 - w), fill=fill)


def draw_tri(d, c, r, fill, outline=(20, 20, 20)):
    x, y = c
    pts = [(x, y - r), (x - r, y + r), (x + r, y + r)]
    d.polygon(pts, fill=fill, outline=outline)


def draw_shape(d, kind, c, r, fill):
    if kind == "circle":
        draw_circle(d, c, r, fill)
    elif kind == "square":
        x, y = c
        d.rectangle((x - r, y - r, x + r, y + r), fill=fill, outline=(20, 20, 20), width=2)
    else:
        draw_tri(d, c, r, fill)


# ----------------------------------------------------------------------------- families
def family_fit_ball(rng, img_dir, uid, foil=False):
    """THE paper example. imagine-track: numbers are in the question."""
    w = rng.randint(6, 18)
    h = rng.randint(6, 18)
    if foil:
        # force the opposite of a typical 'just barely' case
        d = min(w, h) + rng.randint(1, 4) if rng.random() < 0.5 else min(w, h) - rng.randint(0, 2)
        d = max(3, d)
    else:
        d = rng.randint(4, 18)
    fits = (d <= w) and (d <= h)
    q = (f"Can a ball of diameter {d} cm fit inside a box that is {w} cm wide and {h} cm tall?")
    steps_txt = [
        f"Identify the box: width {w} cm, height {h} cm.",
        f"The ball needs {d} cm in every direction.",
        f"Compare: {d} {'≤' if d <= w else '>'} {w} (width) and {d} {'≤' if d <= h else '>'} {h} (height).",
    ]
    # sketches
    def sketch(stage):
        im, dr = canvas()
        # box scaled into the canvas
        scale = 140 / max(w, h)
        bw, bh = w * scale, h * scale
        x0, y0 = 50, 180 - bh
        x1, y1 = x0 + bw, y0 + bh
        draw_rect(dr, (x0, y0, x1, y1), fill=None, outline=(40, 80, 160), w=3)
        label(dr, (x0, y1 + 4), f"{w} cm", size=13)
        label(dr, (8, (y0 + y1) // 2), f"{h}", size=13)
        if stage >= 1:
            r = (d * scale) / 2
            cx = 50 + bw + 28 + r if stage == 1 else (x0 + x1) / 2
            cy = (y0 + y1) / 2 if stage >= 2 else 70
            col = (40, 160, 80) if fits else (200, 60, 60)
            draw_circle(dr, (cx, cy), r, (230, 230, 230), outline=col, w=3)
            label(dr, (cx - 18, cy - r - 16), f"d={d}", fill=col, size=13)
            if stage >= 2 and not fits:
                label(dr, (40, 8), "does not fit", fill=(180, 30, 30), size=16)
            elif stage >= 2 and fits:
                label(dr, (40, 8), "fits", fill=(20, 120, 40), size=16)
        return im

    p0 = save(sketch(0), img_dir, f"{uid}_s0.jpg")
    p1 = save(sketch(1), img_dir, f"{uid}_s1.jpg")
    p2 = save(sketch(2), img_dir, f"{uid}_s2.jpg")
    choices = ["Yes", "No"]
    ans = 0 if fits else 1
    return {
        "id": uid, "family": "fit_ball_box", "track": "imagine", "gate": 1.0,
        "question": q, "choices": choices, "answer": ans, "hint": "",
        "solution": " ".join(steps_txt) + (" Yes." if fits else " No."),
        "has_image": True, "image_path": p2,
        "steps": [
            {"text": steps_txt[0], "image_path": p0, "gate": 0.7},
            {"text": steps_txt[1], "image_path": p1, "gate": 0.9},
            {"text": steps_txt[2], "image_path": p2, "gate": 0.95},
        ],
        "spec": {"w": w, "h": h, "d": d, "fits": fits},
    }


def family_fit_rect(rng, img_dir, uid):
    W, H = rng.randint(8, 20), rng.randint(8, 20)
    w, h = rng.randint(4, 18), rng.randint(4, 18)
    fits = (w <= W) and (h <= H)
    q = (f"Can a {w} by {h} cm rectangle be placed inside a {W} by {H} cm rectangle "
         f"without rotating it?")
    im, dr = canvas()
    s = 150 / max(W, H)
    x0, y0 = 40, 200 - H * s
    draw_rect(dr, (x0, y0, x0 + W * s, y0 + H * s), fill=None, outline=(40, 80, 160), w=3)
    col = (40, 160, 80) if fits else (200, 60, 60)
    draw_rect(dr, (x0 + 6, y0 + 6, x0 + 6 + w * s, y0 + 6 + h * s), fill=None, outline=col, w=2)
    label(dr, (40, 8), "fits" if fits else "does not fit", fill=col, size=16)
    p = save(im, img_dir, f"{uid}.jpg")
    return {
        "id": uid, "family": "fit_rect", "track": "imagine", "gate": 1.0,
        "question": q, "choices": ["Yes", "No"], "answer": 0 if fits else 1,
        "hint": "", "solution": f"Need {w}≤{W} and {h}≤{H}.", "has_image": True, "image_path": p,
        "steps": [{"text": "Draw both rectangles to scale.", "image_path": p, "gate": 0.9}],
        "spec": {"W": W, "H": H, "w": w, "h": h, "fits": fits},
    }


def family_triangle(rng, img_dir, uid):
    a, b, c = sorted(rng.randint(3, 15) for _ in range(3))
    # reshuffle display order
    sides = [a, b, c]
    rng.shuffle(sides)
    a, b, c = sides
    ok = (a + b > c) and (a + c > b) and (b + c > a)
    q = f"Can three sticks of length {a} cm, {b} cm and {c} cm form a triangle?"
    im, dr = canvas()
    if ok:
        # place a triangle with those side lengths (approx, SSS)
        x0, y0 = 40, 200
        x1, y1 = 40 + 12 * a, 200
        # law of cosines for vertex
        cosA = (a * a + b * b - c * c) / (2 * a * b + 1e-6)
        cosA = max(-1, min(1, cosA))
        ang = math.acos(cosA)
        scale = 12
        x2 = 40 + scale * b * math.cos(ang)
        y2 = 200 - scale * b * math.sin(ang)
        dr.polygon([(x0, y0), (x1, y1), (x2, y2)], outline=(40, 80, 160))
        label(dr, (40, 8), "valid triangle", fill=(20, 120, 40), size=16)
    else:
        dr.line((40, 180, 40 + 12 * a, 180), fill=(200, 60, 60), width=3)
        dr.line((40, 200, 40 + 12 * b, 200), fill=(50, 90, 190), width=3)
        dr.line((40, 220, 40 + 12 * c, 220), fill=(40, 150, 80), width=3)
        label(dr, (40, 8), "cannot form a triangle", fill=(180, 30, 30), size=15)
    p = save(im, img_dir, f"{uid}.jpg")
    return {
        "id": uid, "family": "triangle", "track": "imagine", "gate": 1.0,
        "question": q, "choices": ["Yes", "No"], "answer": 0 if ok else 1,
        "hint": "", "solution": f"Triangle inequality on {a},{b},{c}.",
        "has_image": True, "image_path": p,
        "steps": [{"text": "Check a+b>c, a+c>b, b+c>a.", "image_path": p, "gate": 0.85}],
        "spec": {"a": a, "b": b, "c": c, "ok": ok},
    }


def family_compare_len(rng, img_dir, uid):
    a, b = rng.randint(3, 18), rng.randint(3, 18)
    if a == b:
        b = a + rng.choice([-2, -1, 1, 2]); b = max(2, b)
    q = f"Which is longer: a {a} cm rod or a {b} cm rod?"
    im, dr = canvas()
    dr.line((30, 90, 30 + 10 * a, 90), fill=(200, 60, 60), width=8)
    dr.line((30, 160, 30 + 10 * b, 160), fill=(50, 90, 190), width=8)
    label(dr, (30, 60), f"A = {a} cm", fill=(200, 60, 60), size=14)
    label(dr, (30, 170), f"B = {b} cm", fill=(50, 90, 190), size=14)
    p = save(im, img_dir, f"{uid}.jpg")
    choices = ["the  first rod (A)", "the second rod (B)"]
    ans = 0 if a > b else 1
    return {
        "id": uid, "family": "compare_len", "track": "imagine", "gate": 1.0,
        "question": q, "choices": choices, "answer": ans, "hint": "",
        "solution": f"{a} vs {b}.", "has_image": True, "image_path": p,
        "steps": [{"text": "Draw both rods to scale.", "image_path": p, "gate": 0.8}],
        "spec": {"a": a, "b": b},
    }


def _place_objects(rng, n=4, grid=3):
    cells = [(r, c) for r in range(grid) for c in range(grid)]
    rng.shuffle(cells)
    used = []
    taken = set()
    for i in range(n):
        r, c = cells[i]
        color_i = rng.randrange(len(NAMES))
        kind = rng.choice(SHAPES)
        # unique (color, shape) when possible
        key = (color_i, kind)
        if key in taken:
            kind = rng.choice([s for s in SHAPES if (color_i, s) not in taken] or SHAPES)
            key = (color_i, kind)
        taken.add(key)
        used.append({"r": r, "c": c, "color": NAMES[color_i], "rgb": PAL[color_i], "shape": kind})
    return used


def _draw_objects(objs, grid=3, highlight=None):
    im, dr = canvas()
    cell = 70
    ox, oy = 22, 22
    for i in range(grid + 1):
        dr.line((ox, oy + i * cell, ox + grid * cell, oy + i * cell), fill=(210, 210, 210), width=1)
        dr.line((ox + i * cell, oy, ox + i * cell, oy + grid * cell), fill=(210, 210, 210), width=1)
    for o in objs:
        cx = ox + o["c"] * cell + cell // 2
        cy = oy + o["r"] * cell + cell // 2
        draw_shape(dr, o["shape"], (cx, cy), 18, o["rgb"])
        if highlight is not None and o is highlight:
            dr.ellipse((cx - 26, cy - 26, cx + 26, cy + 26), outline=(0, 0, 0), width=2)
    return im


def _obj_name(o):
    return f"the {o['color']} {o['shape']}"


def family_spatial_imagine(rng, img_dir, uid):
    """Text lists every object+cell; question asks a spatial relation. Imagination is possible."""
    grid = 3
    objs = _place_objects(rng, n=rng.randint(3, 5), grid=grid)
    lines = [f"- {_obj_name(o)} is in row {o['r']+1}, column {o['c']+1} (row 1 is top, column 1 is left)."
             for o in objs]
    target = rng.choice(objs)
    mode = rng.choice(["leftmost", "topmost", "color_of_cell", "left_of"])
    if mode == "leftmost":
        gold_o = min(objs, key=lambda o: (o["c"], o["r"]))
        q = "A 3x3 grid contains these objects:\n" + "\n".join(lines) + "\nWhich object is the leftmost?"
        choices = [_obj_name(o)[4:] for o in objs[:4]]
        while len(choices) < 4:
            choices.append(f"{rng.choice(NAMES)} {rng.choice(SHAPES)}")
        ans_s = _obj_name(gold_o)[4:]
        if ans_s not in choices:
            choices[0] = ans_s
        ans = choices.index(ans_s)
    elif mode == "topmost":
        gold_o = min(objs, key=lambda o: (o["r"], o["c"]))
        q = "A 3x3 grid contains these objects:\n" + "\n".join(lines) + "\nWhich object is the topmost?"
        choices = [_obj_name(o)[4:] for o in objs[:4]]
        while len(choices) < 4:
            choices.append(f"{rng.choice(NAMES)} {rng.choice(SHAPES)}")
        ans_s = _obj_name(gold_o)[4:]
        if ans_s not in choices:
            choices[0] = ans_s
        ans = choices.index(ans_s)
        target = gold_o
    elif mode == "color_of_cell":
        q = ("A 3x3 grid contains these objects:\n" + "\n".join(lines) +
             f"\nWhat color is the object in row {target['r']+1}, column {target['c']+1}?")
        choices = []
        for n in NAMES:
            if n not in choices:
                choices.append(n)
        choices = choices[:4]
        if target["color"] not in choices:
            choices[0] = target["color"]
        ans = choices.index(target["color"])
        gold_o = target
    else:
        others = [o for o in objs if o["c"] < target["c"]]
        if not others:
            others = [o for o in objs if o["c"] > target["c"]]
            rel, key = "right", lambda o: -o["c"]
        else:
            rel, key = "left", lambda o: o["c"]
        gold_o = sorted(others, key=key)[0] if others else target
        q = ("A 3x3 grid contains these objects:\n" + "\n".join(lines) +
             f"\nWhich object is to the {rel} of {_obj_name(target)}?")
        choices = [_obj_name(o)[4:] for o in objs[:4]]
        ans_s = _obj_name(gold_o)[4:]
        if ans_s not in choices:
            choices[-1] = ans_s
        ans = choices.index(ans_s)
    im = _draw_objects(objs, grid, highlight=gold_o)
    p = save(im, img_dir, f"{uid}.jpg")
    return {
        "id": uid, "family": "spatial_imagine", "track": "imagine", "gate": 1.0,
        "question": q, "choices": choices, "answer": ans, "hint": "",
        "solution": _obj_name(gold_o), "has_image": True, "image_path": p,
        "steps": [{"text": "Sketch the grid from the description.", "image_path": p, "gate": 0.95}],
        "spec": {"objs": objs, "mode": mode},
    }


def family_count_see(rng, img_dir, uid):
    """SEE track: question does not list the objects. Image is required."""
    n = rng.randint(5, 12)
    objs = []
    for _ in range(n):
        color_i = rng.randrange(len(NAMES))
        objs.append({"color": NAMES[color_i], "rgb": PAL[color_i], "shape": rng.choice(SHAPES),
                     "xy": (rng.randint(30, 226), rng.randint(40, 226))})
    ask_color = rng.choice(NAMES)
    ask_shape = rng.choice(SHAPES)
    gold = sum(1 for o in objs if o["color"] == ask_color and o["shape"] == ask_shape)
    im, dr = canvas()
    for o in objs:
        draw_shape(dr, o["shape"], o["xy"], 14, o["rgb"])
    p = save(im, img_dir, f"{uid}.jpg")
    opts = list(dict.fromkeys([gold, max(0, gold - 1), gold + 1, gold + 2, max(0, gold - 2)]))[:4]
    rng.shuffle(opts)
    q = f"How many {ask_color} {ask_shape}s are in the figure?"
    return {
        "id": uid, "family": "count_see", "track": "see", "gate": 1.0,
        "question": q, "choices": [str(x) for x in opts], "answer": opts.index(gold),
        "hint": "", "solution": str(gold), "has_image": True, "image_path": p,
        "steps": [{"text": "Count the requested shapes in the figure.", "image_path": p, "gate": 1.0}],
        "spec": {"gold": gold, "ask": (ask_color, ask_shape)},
    }


def family_color_see(rng, img_dir, uid):
    """SEE track: one marked object, ask its color/shape. Text has no color word of the answer."""
    objs = _place_objects(rng, n=rng.randint(3, 5), grid=3)
    target = rng.choice(objs)
    im = _draw_objects(objs, 3, highlight=target)
    p = save(im, img_dir, f"{uid}.jpg")
    mode = rng.choice(["color", "shape"])
    if mode == "color":
        q = "The object inside the black ring: what color is it?"
        choices = list(dict.fromkeys([target["color"]] + [rng.choice(NAMES) for _ in range(6)]))[:4]
        ans = choices.index(target["color"])
    else:
        q = "The object inside the black ring: what shape is it?"
        choices = list(dict.fromkeys([target["shape"]] + SHAPES))[:3]
        ans = choices.index(target["shape"])
    return {
        "id": uid, "family": "mark_see", "track": "see", "gate": 1.0,
        "question": q, "choices": choices, "answer": ans, "hint": "",
        "solution": choices[ans], "has_image": True, "image_path": p,
        "steps": [{"text": "Look at the ringed object.", "image_path": p, "gate": 1.0}],
        "spec": {"target": target, "mode": mode},
    }


def family_clock(rng, img_dir, uid):
    h = rng.randint(0, 11)
    m = rng.choice([0, 15, 30, 45])
    hour_ang = 0.5 * (60 * h + m)  # degrees, 0 at 12
    min_ang = 6 * m
    diff = abs(hour_ang - min_ang)
    diff = min(diff, 360 - diff)
    gold = int(round(diff))
    q = f"What is the smaller angle between the hour and minute hands at {h}:{m:02d}?"
    im, dr = canvas()
    cx, cy, r = 128, 128, 90
    draw_circle(dr, (cx, cy), r, (250, 250, 250), outline=(30, 30, 30), w=3)
    def hand(ang_deg, length, w, col):
        rad = math.radians(ang_deg - 90)
        dr.line((cx, cy, cx + length * math.cos(rad), cy + length * math.sin(rad)), fill=col, width=w)
    hand(hour_ang, 50, 6, (200, 60, 60))
    hand(min_ang, 75, 3, (50, 90, 190))
    label(dr, (100, 8), f"{h}:{m:02d}", size=16)
    p = save(im, img_dir, f"{uid}.jpg")
    opts = sorted({gold, (gold + 30) % 180, abs(gold - 30), 90 if gold != 90 else 60})
    opts = [x for x in opts if x <= 180][:4]
    if gold not in opts:
        opts[0] = gold
    rng.shuffle(opts)
    return {
        "id": uid, "family": "clock", "track": "imagine", "gate": 1.0,
        "question": q, "choices": [f"{x} degrees" for x in opts], "answer": opts.index(gold),
        "hint": "", "solution": f"{gold} degrees", "has_image": True, "image_path": p,
        "steps": [{"text": "Sketch a clock and both hands.", "image_path": p, "gate": 0.9}],
        "spec": {"h": h, "m": m, "gold": gold},
    }


ABSTRACT = [
    ("What is the capital of France?", ["Paris", "Lyon", "Marseille", "Nice"], 0),
    ("What is 17 + 24?", ["41", "31", "51", "39"], 0),
    ("Which number is prime?", ["9", "15", "21", "13"], 3),
    ("What is the capital of Japan?", ["Kyoto", "Osaka", "Tokyo", "Nagoya"], 2),
    ("2^5 equals?", ["10", "25", "32", "64"], 2),
    ("Antonym of 'cold'?", ["hot", "freeze", "ice", "winter"], 0),
    ("How many days in a week?", ["5", "6", "7", "8"], 2),
    ("Water chemical formula?", ["CO2", "H2O", "O2", "NaCl"], 1),
]


def family_abstract(rng, img_dir, uid):
    q, ch, ans = ABSTRACT[rng.randrange(len(ABSTRACT))]
    # blank figure — gate should be ~0
    im, dr = canvas()
    label(dr, (40, 110), "(no figure needed)", fill=(140, 140, 140), size=16)
    p = save(im, img_dir, f"{uid}.jpg")
    return {
        "id": uid, "family": "abstract", "track": "abstract", "gate": 0.0,
        "question": q, "choices": ch, "answer": ans, "hint": "",
        "solution": ch[ans], "has_image": False, "image_path": p,
        "steps": [{"text": q, "image_path": p, "gate": 0.05}],
        "spec": {},
    }


FAMILIES_MAIN = [
    (family_fit_ball, 0.18),
    (family_fit_rect, 0.12),
    (family_triangle, 0.12),
    (family_compare_len, 0.08),
    (family_spatial_imagine, 0.18),
    (family_count_see, 0.16),
    (family_color_see, 0.10),
    (family_clock, 0.06),
]


def pick_family(rng):
    u = rng.random()
    acc = 0.0
    for fn, p in FAMILIES_MAIN:
        acc += p
        if u <= acc:
            return fn
    return FAMILIES_MAIN[-1][0]


def make_item(rng, split, i, img_dir):
    fn = pick_family(rng)
    uid = f"{split}_{i:06d}"
    it = fn(rng, img_dir, uid)
    it["split"] = split
    return it


def dump_split(name, n, seed, extra_abstract=0):
    rng = random.Random(seed)
    img_dir = os.path.join(ROOT, "images", name)
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for i in range(n):
        items.append(make_item(rng, name, i, img_dir))
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i+1}/{n}", flush=True)
    abs_items = []
    for j in range(extra_abstract):
        abs_items.append(family_abstract(rng, os.path.join(ROOT, "images", "abstract"), f"{name}_abs_{j:05d}"))
    path = os.path.join(ROOT, f"{name}.json")
    json.dump(items, open(path, "w", encoding="utf-8"))
    if abs_items:
        ap = os.path.join(ROOT, f"{name}_abstract.json")
        json.dump(abs_items, open(ap, "w", encoding="utf-8"))
    fam = Counter(x["family"] for x in items)
    trk = Counter(x["track"] for x in items)
    print(f"[{name}] {len(items)} items -> {path}")
    print("  families:", dict(fam))
    print("  tracks:  ", dict(trk))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_abstract", type=int, default=2000, help="gate=0 items (separate file, not for Phase A)")
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    banner("building spatial CoT dataset (paper: ball-in-box + grid + geometry)")
    dump_split("train", a.n_train, 0)
    dump_split("val", a.n_val, 1, extra_abstract=a.n_abstract)
    dump_split("test", a.n_test, 2)
    man = {"n_train": a.n_train, "n_val": a.n_val, "n_test": a.n_test,
           "note": "Phase A/B/C should use train.json + val.json. abstract files are for the gate only."}
    json.dump(man, open(os.path.join(ROOT, "manifest.json"), "w"), indent=2)
    print("\nexample train question:")
    ex = json.load(open(os.path.join(ROOT, "train.json"), encoding="utf-8"))[0]
    print(" ", ex["question"][:200])
    print("  choices", ex["choices"], "answer", ex["answer"], "family", ex["family"], "track", ex["track"])
    print(f"\nsaved under {ROOT}")
    print("next: python imagine_step8_advanced_a.py --data spatial --vis grid --grid 3 --epochs 3")
    print("BUILD OK")


if __name__ == "__main__":
    main()
