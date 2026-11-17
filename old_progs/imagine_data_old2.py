# Shared helpers: banners, seeds, dataset download + caching (COCO, ScienceQA), CLIP embeddings.
# Not run directly. Imported by the step scripts.
import os, io, re, json, random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

CACHE, CKPT, OUT = "imagine_cache", "imagine_ckpt", "imagine_out"
for _d in (CACHE, CKPT, OUT):
    os.makedirs(_d, exist_ok=True)

def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def split_steps(text, max_steps=4):
    """Split a ScienceQA 'solution' into short reasoning steps (sentences)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 8]
    return sents[:max_steps]

# ----------------------------------------------------------------------------- COCO
COCO_SOURCES = [  # tried in order until one works; all are streamed (no giant download)
    ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", "test"),
    ("lmms-lab/COCO-Caption", "val"),
    ("yerevann/coco-captions", "train"),
]

def _norm_caps(v):
    if v is None: return []
    if isinstance(v, str): return [v.strip()]
    out = []
    for x in v:
        if isinstance(x, str): out.append(x)
        elif isinstance(x, dict): out.append(x.get("raw") or x.get("text") or "")
    return [c.strip() for c in out if c and c.strip()]

def _get_image(row):
    for v in row.values():
        if isinstance(v, Image.Image): return v
    url = row.get("url") or row.get("coco_url")
    if isinstance(url, str) and url.startswith("http"):
        import requests
        r = requests.get(url, timeout=10); r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    return None

def get_coco(n=2000):
    """Returns [{'path': jpg_path, 'captions': [str,...]}, ...] (cached on disk)."""
    meta_path = f"{CACHE}/coco_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if len(meta) >= n and all(os.path.exists(m["path"]) for m in meta[:n]):
            print(f"[data] COCO cache hit: {n} images", flush=True); return meta[:n]
    from datasets import load_dataset
    os.makedirs(f"{CACHE}/coco_images", exist_ok=True)
    last_err = None
    for name, split in COCO_SOURCES:
        try:
            print(f"[data] trying {name} ({split}) ...", flush=True)
            ds = load_dataset(name, split=split, streaming=True)
            meta, pbar = [], tqdm(total=n, desc=f"COCO <- {name.split('/')[-1]}")
            for row in ds:
                caps = []
                for k in ("captions", "caption", "sentences", "answer", "text"):
                    if k in row:
                        caps = _norm_caps(row[k])
                        if caps: break
                if not caps: continue
                try: img = _get_image(row)
                except Exception: continue
                if img is None: continue
                img = img.convert("RGB"); img.thumbnail((256, 256))
                p = f"{CACHE}/coco_images/{len(meta):05d}.jpg"; img.save(p, quality=90)
                meta.append({"path": p, "captions": caps[:5]}); pbar.update(1)
                if len(meta) >= n: break
            pbar.close()
            if len(meta) >= max(200, n // 2):
                json.dump(meta, open(meta_path, "w")); print(f"[data] saved {len(meta)} COCO images", flush=True)
                return meta
            print(f"[data] only {len(meta)} usable rows from {name}, trying next source", flush=True)
        except Exception as e:
            last_err = e; print(f"[data] {name} failed: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError(f"Could not download any COCO source. Last error: {last_err}\n"
                       "Check internet / `pip install -U datasets` and send me this error.")

# ----------------------------------------------------------------------------- ScienceQA
def get_scienceqa(n_img=400, n_noimg=400, min_steps=0, max_scan=40000):
    """ScienceQA items; n_img with an image, n_noimg without. Cached on disk."""
    path = f"{CACHE}/sqa_{n_img}_{n_noimg}_{min_steps}.json"
    if os.path.exists(path):
        items = json.load(open(path)); print(f"[data] ScienceQA cache hit: {len(items)} items", flush=True); return items
    from datasets import load_dataset
    os.makedirs(f"{CACHE}/sqa_images", exist_ok=True)
    print("[data] streaming derek-thomas/ScienceQA ...", flush=True)
    ds = load_dataset("derek-thomas/ScienceQA", split="train", streaming=True)
    items, ci, cn = [], 0, 0
    pbar = tqdm(total=n_img + n_noimg, desc="ScienceQA")
    for k, row in enumerate(ds):
        if k >= max_scan or (ci >= n_img and cn >= n_noimg): break
        has = row.get("image") is not None
        if has and ci >= n_img: continue
        if (not has) and cn >= n_noimg: continue
        sol = (row.get("solution") or "").strip()
        if len(split_steps(sol)) < min_steps: continue
        it = {"question": row["question"], "choices": list(row["choices"]), "answer": int(row["answer"]),
              "hint": (row.get("hint") or "").strip(), "solution": sol, "has_image": bool(has), "image_path": None}
        if has:
            img = row["image"].convert("RGB"); img.thumbnail((256, 256))
            it["image_path"] = f"{CACHE}/sqa_images/{len(items):05d}.jpg"; img.save(it["image_path"], quality=90)
            ci += 1
        else:
            cn += 1
        items.append(it); pbar.update(1)
    pbar.close()
    print(f"[data] ScienceQA collected: {ci} with image, {cn} without (scanned {k} rows)", flush=True)
    random.Random(0).shuffle(items)
    json.dump(items, open(path, "w"))
    return items

def get_spatial_cot(split="train", n=None):
    """Items from imagine_build_spatial_cot.py. split: train | val | test | validation."""
    name = "val" if split in ("val", "validation") else split
    path = f"{CACHE}/spatial_cot/{name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python imagine_build_spatial_cot.py --n_train 20000 --n_val 2000 --n_test 2000")
    items = json.load(open(path, encoding="utf-8"))
    if n is not None:
        items = items[:n]
    missing = [i["image_path"] for i in items[:20] if not os.path.exists(i["image_path"])]
    if missing:
        raise FileNotFoundError(f"spatial_cot images missing e.g. {missing[0]}. Re-run imagine_build_spatial_cot.py")
    print(f"[data] spatial_cot[{name}] {len(items)} items", flush=True)
    return items

# ----------------------------------------------------------------------------- CLIP
_clip = None
def load_clip():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("[load] CLIP ViT-B/32 (512-d)", flush=True)
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(dev).float().eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip = (m, p, dev)
    return _clip

def _feat(e):
    assert torch.is_tensor(e) and e.shape[-1] == 512, f"unexpected CLIP output {type(e)}"
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu()

@torch.no_grad()
def clip_image_embs(paths, bs=32):
    m, p, dev = load_clip(); outs = []
    for i in tqdm(range(0, len(paths), bs), desc="CLIP image emb"):
        ims = [Image.open(x).convert("RGB") for x in paths[i:i + bs]]
        outs.append(_feat(m.get_image_features(**p(images=ims, return_tensors="pt").to(dev))))
    return torch.cat(outs)

@torch.no_grad()
def clip_text_embs(texts, bs=128):
    m, p, dev = load_clip(); outs = []
    for i in tqdm(range(0, len(texts), bs), desc="CLIP text emb"):
        inp = p(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        outs.append(_feat(m.get_text_features(**inp)))
    return torch.cat(outs)
