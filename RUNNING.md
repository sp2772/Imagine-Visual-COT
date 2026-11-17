# Running the Project

> [!IMPORTANT]
> **Always run every script from the project root directory.**
> This ensures that cross-script imports and all data paths (`imagine_cache/`, `imagine_ckpt/`, `imagine_out/`) resolve correctly without any path configuration.

```powershell
# Always start here:
cd path/to/Imagine-Visual-COT
```

---

## Environment Setup

```powershell
pip install -r requirements.txt
```

**Hardware used:** Windows, NVIDIA RTX 3050 6 GB VRAM, 24 GB RAM, Python 3.9.

> [!TIP]
> Run `python dataset/imagine_check_env.py` first to verify all library versions are consistent before running any experiment.

---

## Step 0 — Build the Dataset

```powershell
# Generate the leak-free spatial dataset (~10 min, creates imagine_cache/)
python dataset/imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000

# Audit for data leaks (should report 0/8 families flagged)
python dataset/imagine_audit_data.py imagine_cache/spatial_cot_v2/val.json
```

---

## Track 1 — SmolLM2-135M (Steps 1–8d)

Early-stage experiments building the core modules one by one. Each script is self-contained and runs in a few minutes.

```powershell
python experiments/track1_smollm2/imagine_step1.py    # Load backbone, extract h_t
python experiments/track1_smollm2/imagine_step2.py    # Imagination Head + gate
python experiments/track1_smollm2/imagine_step3.py    # Anti-collapse loss demo
python experiments/track1_smollm2/imagine_step4.py    # Feedback chain (causality check)
python experiments/track1_smollm2/imagine_step5.py    # Stage 1: COCO alignment (top-1 → 0.34)
python experiments/track1_smollm2/imagine_step6.py    # Gate training (val acc 0.956)
python experiments/track1_smollm2/imagine_step7.py    # Retrieval decoder gallery
python experiments/track1_smollm2/imagine_step8.py    # Joint training (LM loss 1.65 → 1.33)
python experiments/track1_smollm2/imagine_step8b.py   # Soft-prompt control experiment
python experiments/track1_smollm2/imagine_step8c.py   # Teacher forcing + LoRA
python experiments/track1_smollm2/imagine_step8d.py   # Bootstrap CI significance testing
```

Steps 9 and 10 (LoRA joint + GRPO) are written but were never run:
```powershell
python experiments/track1_smollm2/imagine_step9.py    # (not yet run)
python experiments/track1_smollm2/imagine_step10.py   # (not yet run)
```

---

## Track 2 — Qwen2.5-0.5B Advanced Track (Phases A0 → A → B → C)

The main result track. Run phases in order. Each phase builds on the previous checkpoint.

### Phase A0 — Caption Alignment (~8 min)

Aligns the Reader on 20 k COCO image–caption pairs. Run once per grid size.

```powershell
python experiments/track2_advanced/imagine_step8_advanced_a0.py --vis grid --grid 3
```

### Phase A — Reader + LoRA (~40 min/epoch × 4 epochs)

Trains the Reader and LoRA adapters on the spatial answer-first task.

```powershell
python experiments/track2_advanced/imagine_step8_advanced_a.py \
    --data spatial2 --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4
```

**Interrupted?** Add `--resume`.  
**Out of memory?** Add `--micro_bs 4 --accum 4`.

### Phase A Evaluation (no training)

Check oracle / wrong / blank accuracy on any split without training.

```powershell
python experiments/track2_advanced/imagine_step8_advanced_a_eval.py \
    --data spatial2 --vis grid --grid 3 --split test_ood --n 2000
```

### Phase B — Imagination Head (~17 min/epoch × 3 epochs)

Trains the Imagination Head while Reader + LoRA + LLM are frozen.

```powershell
python experiments/track2_advanced/imagine_step8_advanced_b.py \
    --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3
```

### Phase C — Held-out Evaluation

Produces per-family accuracy tables and retrieval gallery PNGs in `imagine_out/`.

```powershell
# In-distribution test
python experiments/track2_advanced/imagine_step8_advanced_c.py \
    --data spatial2 --vis grid --grid 3 --track imagine --split test --n_test 2000

# Out-of-distribution test
python experiments/track2_advanced/imagine_step8_advanced_c.py \
    --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000
```

---

## Grid 7 Variant (7×7 = 49 patch tokens)

Requires running A0 for grid 7 first — skipping A0 caused the inconclusive earlier run.

```powershell
# Step 1: A0 alignment for grid 7 (~35–45 min)
python experiments/track2_advanced/imagine_step8_advanced_a0.py --vis grid --grid 7

# Step 2: Phase A with grid 7 (~85 min)
python experiments/track2_advanced/imagine_step8_advanced_a.py \
    --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4
```

---

## Dataset Utilities

```powershell
# Diagnose ScienceQA text-solvability
python dataset/imagine_diagnose_data.py

# Fetch public datasets (not used in final pipeline)
python dataset/imagine_fetch_public.py

# Build the leaky v1 spatial dataset (superseded by v2, kept for reference)
python dataset/imagine_build_spatial_cot.py --n_train 80000 --n_val 4000 --n_test 4000
```

---

## Output Locations

| Path | Contents |
|------|----------|
| `imagine_cache/` | CLIP features, spatial v2 JSON + images (~4.2 GB, gitignored) |
| `imagine_ckpt/` | All model checkpoints (tracked in git) |
| `imagine_out/` | Result gallery PNGs from Phase C and Step 7 |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: upgrade torch to at least v2.6` | Old torch refusing `.bin` CLIP weights | Already fixed in `lib/imagine_data.py` via `use_safetensors=True` |
| `Dataset scripts are no longer supported` | Deprecated HuggingFace source | Loader falls back to `lmms-lab/COCO-Caption` automatically |
| `NameError: suffix_of` or `TypeError: root` | Stale `lib/` files vs script version | Run `python dataset/imagine_check_env.py`; replace outdated files |
| Grid-7 Phase A not converging | A0 alignment was skipped | Run `imagine_step8_advanced_a0.py --grid 7` first |
| OOM on 6 GB GPU | Batch size too large | Add `--micro_bs 4 --accum 4` |
