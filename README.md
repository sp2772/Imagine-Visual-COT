# Imagine: Latent Visual Imagination for LLM Reasoning

> **Can a language model reason better by *picturing* the problem?**
>
> This project implements and rigorously evaluates an **Imagination Head** — a module that lets an LLM produce a latent visual representation at every reasoning step, using images only as a training-time teacher signal. At test time the model is a *blank-slate student*: no image is ever provided; it must imagine by itself.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Architecture](#2-architecture)
3. [Key Results](#3-key-results)
4. [Project Structure](#4-project-structure)
5. [Quickstart](#5-quickstart)
6. [Training Pipeline](#6-training-pipeline)
7. [Dataset: Spatial CoT v2](#7-dataset-spatial-cot-v2)
8. [Evaluation Conditions](#8-evaluation-conditions)
9. [Findings Summary](#9-findings-summary)
10. [Environment & Hardware](#10-environment--hardware)
11. [Planned Next Steps](#11-planned-next-steps)
12. [Citation / Reference](#12-citation--reference)

---

## 1. The Core Idea

Language models (LLMs) reason step-by-step in text ("chain-of-thought"). Humans also *visualize* problems. The original research plan (`research_plan_new.pdf`) proposes adding an **internal imagination** to an LLM:

```
                 question text
                       │
                       ▼
        ┌──────────────────────────────┐
        │  LLM (Qwen2.5-0.5B + LoRA)  │──▶ hidden state h  (end of question)
        └──────────────────────────────┘           │
                 ▲         │                       ▼
                 │         │             Imagination Head (MLP)
                 │         │                       │
                 │         │                       ▼
                 │         │          z = 3×3 grid of 768-d tokens
                 │         │               ("imagined picture")
            visual tokens  │                       │
                 ▲         │                       │
                 └──── Reader (MLP) ◀──────────────┘
                           │
                           ▼
                  "Answer:" → letter A/B/C/D

 TEACHER (training only): true image → CLIP → 3×3×768 → Reader → LLM
 TEST: no image anywhere; only the student's own imagination ("own").
```

The key design principle: **images are a teaching signal only**, analogous to teacher forcing in next-token training.

---

## 2. Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | Qwen2.5-0.5B (advanced track) · SmolLM2-135M (early steps) |
| **LoRA** | r=16 adapters, ~2.16 M trainable params of 496 M total |
| **Imagination Head** | 3-layer MLP: 896 → 2048 → 2048 → 6912 · **20.2 M params** |
| **Reader** | LayerNorm + MLP + position embedding; maps 9×768 CLIP patch tokens → 9×896 LLM tokens |
| **CLIP** | ViT-B/32 (frozen); penultimate patch features pooled to 3×3 grid (9 tokens × 768-d) |
| **Answer format** | Answer-first: `[question + choices] → [visual tokens] → "Answer:" → letter` |

### Training Phases

```
A0  Caption alignment  (Reader on 20 k COCO, LLM frozen)   ~8 min
 A  Reader + LoRA      (on spatial answer-first task)        ~40 min/epoch
 B  Imagination Head   (Reader + LoRA + LLM all frozen)      ~17 min/epoch × 3
 C  Evaluation         (held-out test + test_ood, no training)
```

For the full architecture diagrams (Mermaid flowcharts), see **[ARCHITECTURE_MERMAID.md](ARCHITECTURE_MERMAID.md)**.

---

## 3. Key Results

### Phase A — Can the LLM use a real image? ✅

| Condition | Accuracy (val) |
|-----------|---------------|
| oracle (true image) | **81.8%** |
| wrong image | 73.8% |
| blank (text only) | 76.9% |

**oracle vs wrong: +8.1 pp (CI +6.4 to +9.9) — SUPPORTED.** The visual path works.

### Phase B + C — Can the model imagine by itself?

**In-distribution (test, n=1 409)**

| Condition | Accuracy |
|-----------|---------|
| oracle | 97.3% |
| **own** (imagined) | **96.7%** |
| const (blank) | 96.5% |
| own_shuffled | 94.7% |
| wrong | 91.7% |

> **own beats own_shuffled by +1.99 pp (CI +1.21 to +2.84) — SUPPORTED.**
> The model's imagination is **example-specific**: it retrieves the correct diagram in the Top-5 **23.0%** of the time (nearly **60× chance**).
>
> However, `own` ties `const` overall because text-only already saturates most in-distribution families (fit_ball_box, fit_rect, spatial_grid all at 100%).

**Out-of-distribution (test_ood, n=2 000)**

| Condition | Accuracy |
|-----------|---------|
| oracle | 93.7% |
| **own** | **85.9%** |
| const (blank) | 85.3% |
| own_shuffled | 85.1% |
| wrong | 82.8% |

> On OOD `triangle` specifically: **oracle 98.2% vs own 51.7% ≈ const 52.0%** (chance).
> This pinpoints the **single-shot bottleneck**: a 3-layer MLP reading the LLM's final hidden state `h` cannot construct a scale-free geometric sketch when `h` itself failed the OOD arithmetic. One-shot imagination re-represents what `h` already computed.

### Retrieval (Phase C)

| Metric | Score | Chance |
|--------|-------|--------|
| Top-1 | 6.0% | 0.07% |
| Top-5 | 23.0% | 0.36% |
| Median rank | — | ~700 |

---

## 4. Project Structure

```
Imagine_Visual_COT/
├── imagine_step1.py               # Step 1: load backbone, extract h_t
├── imagine_step2.py               # Step 2: Imagination Head + gate
├── imagine_step3.py               # Step 3: anti-collapse loss demo
├── imagine_step4.py               # Step 4: feedback chain (causal, gradients)
├── imagine_step5.py               # Step 5: Stage 1 COCO alignment (SmolLM2)
├── imagine_step6.py               # Step 6: gate training
├── imagine_step7.py               # Step 7: retrieval decoder (stand-in)
├── imagine_step8.py               # Step 8: joint training (SmolLM2, frozen LLM)
├── imagine_step8b.py              # Step 8b: soft-prompt control
├── imagine_step8c.py              # Step 8c: teacher forcing + LoRA
├── imagine_step8d.py              # Step 8d: bootstrap CIs, 150 fresh questions
├── imagine_step8_advanced_a0.py   # Phase A0: caption alignment (Qwen2.5-0.5B)
├── imagine_step8_advanced_a.py    # Phase A: Reader + LoRA on spatial task
├── imagine_step8_advanced_a_eval.py # Phase A evaluation (no training)
├── imagine_step8_advanced_b.py    # Phase B: train Imagination Head
├── imagine_step8_advanced_c.py    # Phase C: held-out evaluation + galleries
├── imagine_adv_lib.py             # Shared library: Reader, Head, eval, bootstrap
├── imagine_data.py                # Data helpers: COCO, ScienceQA, CLIP
├── imagine_build_spatial_v2.py    # Build the leak-free spatial dataset (v2)
├── imagine_build_spatial_cot.py   # Build spatial v1 (leaky — superseded)
├── imagine_audit_data.py          # Data-leak auditor (8 checks per family)
├── imagine_diagnose_data.py       # ScienceQA text-solvability diagnostics
├── imagine_check_env.py           # Version-consistency check
├── imagine_fetch_public.py        # Public dataset fetcher (unused)
├── imagine_step9.py               # (written, never run) LoRA joint
├── imagine_step10.py              # (written, never run) GRPO
│
├── imagine_ckpt/                  # Model checkpoints (tracked)
│   ├── adv_b_qwen2.5-0.5b_grid3_spatial2_imagine/   ← Phase B head (latest)
│   ├── adv_a_qwen2.5-0.5b_grid3_spatial2/           ← Phase A reader + LoRA
│   ├── adv_a0_qwen2.5-0.5b_grid3/                   ← A0 aligned reader
│   ├── lora_step8c/                                  ← Step 8c LoRA adapter
│   ├── chain_step8c.pt                               ← Step 8c chain weights
│   └── ...
│
├── imagine_out/                   # Output galleries (PNG)
│   ├── adv_c_..._test_imagination.png
│   ├── adv_c_..._test_ood_imagination.png
│   └── step7_retrieval.png
│
├── imagine_cache/                 # (NOT tracked — regenerate) CLIP features,
│                                  #   spatial v2 JSON + images (~4.2 GB)
│
├── old_progs/                     # Superseded versions of library scripts
├── ARCHITECTURE_MERMAID.md        # 12 Mermaid diagrams of the full pipeline
├── RESEARCH_SUMMARY_v4.md         # Full lab notebook with all results
├── IMAGINE_RESEARCH_SUMMARY.md    # Earlier summary (v1)
├── IMAGINE_RESEARCH_SUMMARY_v2.md # Earlier summary (v2)
├── RESEARCH_SUMMARY_v3.md         # Earlier summary (v3)
├── research_plan_new.pdf          # Original research plan (PDF)
├── requirements.txt
└── .gitignore
```

---

## 5. Quickstart

> **Always run scripts from the project root** so that cross-module imports and data paths resolve correctly. See **[RUNNING.md](RUNNING.md)** for the full guide.

```powershell
# 1. Clone
git clone https://github.com/sp2772/Imagine-Visual-COT.git
cd Imagine-Visual-COT

# 2. Install dependencies (Python 3.9+, CUDA recommended)
pip install -r requirements.txt

# 3. Sanity check environment
python dataset/imagine_check_env.py

# 4. Build the leak-free spatial dataset (~10 min, creates imagine_cache/)
python dataset/imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000

# 5. Audit the dataset for leaks
python dataset/imagine_audit_data.py imagine_cache/spatial_cot_v2/val.json
```

---

## 6. Training Pipeline

Run these in order. Each script is self-contained with CLI flags.

```powershell
# ── Phase A0: align the reader on COCO captions (once per grid size) ─────────
python imagine_step8_advanced_a0.py --vis grid --grid 3

# ── Phase A: train Reader + LoRA on the spatial task ─────────────────────────
python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 3 \
    --n_train 20000 --n_val 2000 --epochs 4

# ── (Optional) evaluate Phase A without training ──────────────────────────────
python imagine_step8_advanced_a_eval.py --data spatial2 --vis grid --grid 3 \
    --split test_ood --n 2000

# ── Phase B: train the Imagination Head (Reader + LoRA frozen) ───────────────
python imagine_step8_advanced_b.py --data spatial2 --vis grid --grid 3 \
    --track imagine --n_train 20000 --n_val 2000 --epochs 3

# ── Phase C: held-out evaluation + retrieval gallery ────────────────────────
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 \
    --track imagine --split test     --n_test 2000
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 \
    --track imagine --split test_ood --n_test 2000
```

**Out-of-memory?** Add `--micro_bs 4 --accum 4`.  
**Interrupted Phase A?** Add `--resume`.

### Early-stage experiments (SmolLM2-135M, Steps 1–8d)

```powershell
python experiments/track1_smollm2/imagine_step1.py   # backbone + hidden states
python experiments/track1_smollm2/imagine_step2.py   # Imagination Head + gate
python experiments/track1_smollm2/imagine_step3.py   # anti-collapse loss demo
python experiments/track1_smollm2/imagine_step4.py   # feedback loop (causality check)
python experiments/track1_smollm2/imagine_step5.py   # COCO alignment (top-1 0.34)
python experiments/track1_smollm2/imagine_step6.py   # gate training (val acc 0.956)
python experiments/track1_smollm2/imagine_step7.py   # retrieval decoder gallery
python experiments/track1_smollm2/imagine_step8.py   # joint training
python experiments/track1_smollm2/imagine_step8b.py  # soft-prompt control
python experiments/track1_smollm2/imagine_step8c.py  # teacher forcing + LoRA
python experiments/track1_smollm2/imagine_step8d.py  # bootstrap CI significance
```

---

## 7. Dataset: Spatial CoT v2

The `imagine_cache/` folder (not tracked in git) holds the generated spatial dataset. Regenerate it with:

```powershell
python dataset/imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000
```

### Task families

| Track | Family | What it tests |
|-------|--------|--------------|
| `imagine` | `fit_ball_box` | Does a ball fit in a box? (radii) |
| `imagine` | `fit_rect` | Does a rectangle fit inside another? |
| `imagine` | `triangle` | Do three lengths form a valid triangle? |
| `imagine` | `spatial_grid` | Object position on a labeled grid |
| `imagine` | `grid_walk` | Multi-step board state tracking |
| `see` | `count_see` | Count objects in the image |
| `see` | `mark_see` | Identify the ringed object |
| `see` | `rel_see` | Spatial relation (left-of / above) |

**`imagine` track:** the question text fully specifies the scene — a "blank-slate student" can imagine the picture.  
**`see` track:** the image is *necessary*; text alone is uninformative.  
**`test_ood`:** harder held-out split (bigger numbers, larger grids, longer walks).

### Leak-free design rules (v2)

The first dataset version (`imagine_build_spatial_cot.py`, built by an AI coding assistant) had **8/8 families flagged** by the auditor — answer text written in images, green/red colour coding, answer always at index 0, etc. Version 2 enforces:

- Picture shows the **scene or state**, never the verdict
- Yes/no **balanced by construction**
- Choices **shuffled**; answer index **uniform across A–D**
- Distractors **independent** of the gold answer

---

## 8. Evaluation Conditions

Every model is evaluated under five conditions on the same questions:

| Condition | Visual input | Purpose |
|-----------|-------------|---------|
| `oracle` | True image → CLIP → Reader | Upper bound |
| `wrong` | Another question's image | Does image *content* matter? |
| `const` | Learned null tokens | Text-only baseline |
| **`own`** | **Imagination Head output** | **The real test** |
| `own_shuffled` | Another question's imagination | Is imagination example-specific? |

All comparisons use **paired bootstrap 95% CIs** (10 000 resamples). Verdict: **SUPPORTED** (CI > 0), **OPPOSITE** (CI < 0), or *not distinguishable from 0*.

---

## 9. Findings Summary

| # | Finding | Status |
|---|---------|--------|
| 1 | Early "imagination helps" numbers were a **soft-prompt artefact** — a trained extra token, not visual content | ✅ Solid |
| 2 | ScienceQA is **text-solvable**: 65% have hints, 14% contain the gold answer string; swapping the image changes nothing | ✅ Solid |
| 3 | Spatial dataset v1 **leaked the answer** in images (verdict text, green/red colour, answer always at index 0) | ✅ Solid |
| 4 | Leak-free v2: **true image beats wrong image by +8.1 pp (CI +6.4–+9.9)** — the visual path works | ✅ Solid |
| 5 | Text alone already solves most `imagine` families in-distribution (~95–100%); headroom is in `grid_walk` and OOD `triangle` | ✅ Solid |
| 6 | **OOD `triangle`**: text drops to chance (52%), true sketch gives 98% — a scale-free picture generalises where text arithmetic fails | ✅ Solid observation |
| 7 | **Phase B/C in-distribution**: own beats own_shuffled by **+1.99 pp (SUPPORTED)**; Top-5 retrieval 23% (60× chance) — imagination is **example-specific** | ✅ Solid |
| 8 | **Phase C OOD — single-shot bottleneck**: own (85.9%) ≈ const (85.3%). On OOD triangle: own 51.7% = const 52.0% ≪ oracle 98.2%. A one-shot MLP cannot simulate geometry that the LLM's hidden state `h` itself failed to compute | ✅ Solid |

---

## 10. Environment & Hardware

| Item | Value |
|------|-------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3050 6 GB VRAM |
| RAM | 24 GB |
| Python | 3.9 |
| PyTorch | ≥ 2.1 (bf16 + gradient checkpointing) |
| Backbone | Qwen2.5-0.5B (advanced), SmolLM2-135M (early steps) |
| CLIP | ViT-B/32 via `open-clip-torch` |
| LoRA | r=16, via `peft` |

**Known environment fixes** (see Section 11 of `RESEARCH_SUMMARY_v4.md`):
- Load CLIP with `use_safetensors=True` (required for PyTorch < 2.6)
- Use `lmms-lab/COCO-Caption` (the `nlphuji` source is deprecated)
- Run `imagine_check_env.py` after any file update to catch version mismatches

---

## 11. Planned Next Steps

1. **Multi-step `grid_walk` chain** — implement the per-step PDF feedback loop: `z_{t-1} + move_t → z_t`. This is the highest-priority experiment because `grid_walk` has clear headroom (~86–90% in-distribution, ~76% OOD) where even the oracle starting-board image didn't help the static setup.
2. **Grid 7** (7×7 = 49 patch tokens) — after a proper A0 alignment; expected to fix `mark_see` and `rel_see` which were at chance under 3×3 pooling.
3. **Co-adapt the reader in Phase B** — unfreeze the reader at a tiny lr (5e-5) or add a gate blending `own` with `const` to prevent CE degradation on OOD.
4. **Photo VQA (A-OKVQA)** as a natural-image domain.
5. **Gate with an empirical target**: train `g_t` to predict benefit = CE(const) − CE(own) > 0, then blend tokens: `v = g·own + (1−g)·const`.
6. **GRPO** (steps 9–10) adapted to the advanced answer-first track.
7. **Dual-stack architecture**: text stack + latent-image stack with bidirectional cross-attention at each layer, overcoming the single-shot MLP bottleneck.

---

## 12. Citation / Reference

This project implements and extends the ideas from the author's research plan:

```
research_plan_new.pdf — Latent Visual Imagination for LLM Reasoning
```

If you use this codebase or build on these results, please link back to this repository.

---

<details>
<summary><strong>Quick command cheat-sheet</strong></summary>

```powershell
# Check environment consistency
python imagine_check_env.py

# Build + audit dataset
python dataset/imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000
python imagine_audit_data.py imagine_cache/spatial_cot_v2/val.json

# Full advanced-track pipeline
python imagine_step8_advanced_a0.py --vis grid --grid 3
python imagine_step8_advanced_a.py  --data spatial2 --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4
python imagine_step8_advanced_b.py  --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3
python imagine_step8_advanced_c.py  --data spatial2 --vis grid --grid 3 --track imagine --split test     --n_test 2000
python imagine_step8_advanced_c.py  --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000

# Proper Grid 7 (run A0 for grid 7 first!)
python imagine_step8_advanced_a0.py --vis grid --grid 7
python imagine_step8_advanced_a.py  --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4
```

</details>
