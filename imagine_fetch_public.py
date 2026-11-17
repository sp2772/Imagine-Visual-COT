# Optional public datasets from research_plan_new.pdf Stage 1–2.
# Default is cheap: more COCO captions + Visual-CoT JSON (no 143GB images).
#
#   python imagine_fetch_public.py --coco 20000
#   python imagine_fetch_public.py --viscot_json          # ~345 MB annotations only
#   python imagine_fetch_public.py --viscot_images        # WARNING: ~143 GB, only if you mean it
#
# Visual CoT (Shao et al. NeurIPS 2024) is Stage 2 of the PDF: bbox-grounded reasoning
# on REAL images. It does NOT replace the ball-in-box spatial set (Stage 3 analogue).
import argparse, os
from imagine_data import CACHE, banner, get_coco

def fetch_viscot_json():
    from huggingface_hub import hf_hub_download
    banner("Visual-CoT annotations (viscot_363k.json, ~345MB)")
    p = hf_hub_download(repo_id="deepcs233/Visual-CoT", filename="viscot_363k.json",
                        repo_type="dataset", local_dir=f"{CACHE}/viscot")
    print("saved", p)
    return p

def fetch_viscot_images():
    from huggingface_hub import snapshot_download
    banner("Visual-CoT IMAGES (~143GB). This will take a long time.")
    p = snapshot_download(repo_id="deepcs233/Visual-CoT", repo_type="dataset",
                          local_dir=f"{CACHE}/viscot_full",
                          allow_patterns=["cot_images_tar_split/*", "viscot_363k.json", "metadata/*"])
    print("saved", p)
    print("extract the tar parts inside cot_images_tar_split before using.")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=int, default=0, help="download/cache this many COCO caption images")
    ap.add_argument("--viscot_json", action="store_true")
    ap.add_argument("--viscot_images", action="store_true")
    a = ap.parse_args()
    if not (a.coco or a.viscot_json or a.viscot_images):
        print("pass at least one of --coco N  --viscot_json  --viscot_images")
        print("example: python imagine_fetch_public.py --coco 20000 --viscot_json")
        return
    if a.coco:
        banner(f"COCO captions n={a.coco}")
        meta = get_coco(a.coco)
        print(f"COCO ready: {len(meta)} images under {CACHE}/coco_images")
    if a.viscot_json:
        fetch_viscot_json()
    if a.viscot_images:
        fetch_viscot_images()
    print("FETCH OK")

if __name__ == "__main__":
    main()
