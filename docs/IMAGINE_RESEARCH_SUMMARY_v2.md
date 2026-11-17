# Latent Visual Imagination for LLM Reasoning: Project Summary and Lab Notebook

*Originally written 2026-10-24; additively updated 2026-10-24 after Phase B and Phase C (test & test_ood) finished. Covers the whole project from the first message (about 2026-10-20) to the completion of the single-step advanced track.*
*Audience: someone who has never heard of this project. Every term used is explained in the glossary (Section 2).*

---

## Table of contents
1. [What this project is, in 60 seconds](#1-what-this-project-is-in-60-seconds)
2. [Glossary of keywords and notation](#2-glossary-of-keywords-and-notation)
3. [How to read the result tables](#3-how-to-read-the-result-tables)
4. [The idea and how our experiments differ from the original PDF](#4-the-idea-and-how-our-experiments-differ-from-the-original-pdf)
5. [Timeline of what was done, with every intermediate result](#5-timeline-of-what-was-done-with-every-intermediate-result)
6. [Programs inventory (what each file does)](#6-programs-inventory)
7. [Datasets used or built](#7-datasets-used-or-built)
8. [Consolidated results](#8-consolidated-results)
9. [What we have learned (and what we have NOT shown)](#9-what-we-have-learned-and-what-we-have-not-shown)
10. [File versions, `old_progs`, and the run log](#10-file-versions-old_progs-and-the-run-log)
11. [Environment problems met and how they were fixed](#11-environment-problems-met-and-how-they-were-fixed)
12. [Current status and next steps](#12-current-status-and-next-steps)
13. [Command cheat-sheet](#13-command-cheat-sheet)

---

## 1. What this project is, in 60 seconds

**The question.** Today's language models (LLMs) solve problems by writing text step by step ("chain-of-thought"). Humans often also *picture* the problem ("can this ball fit in that box?"). The research plan (`research_plan_new.pdf`) asks: **can we add an internal "imagination" to an LLM**, a learned visual representation produced at every reasoning step, so that its reasoning gets richer and more interpretable?

**The proposed mechanism (from the PDF).**
- A normal LLM backbone reads the problem and produces a hidden state `h_t` at each reasoning step `t`.
- An **Imagination Head** turns `h_t` into a visual vector `z_t = W2·h_t` ("what the model is imagining").
- A **Gate** `g_t = sigmoid(W1·h_t)` (a number from 0 to 1) decides whether the imagination is shown as an image (high) or stays an internal thought (low).
- The previous imagination `z_{t-1}` is **fed back** into the LLM at the next step (a loop).
- A **contrastive anti-collapse loss** forces `z_t` to differ from `z_{t-1}`, because earlier latent-visual-reasoning work reported that imagined tokens collapse into identical vectors.
- Training in four stages: COCO captions, Visual CoT, ScienceQA (plus a GRPO reinforcement step), then joint training. Everything is done with LoRA in bf16 to save memory.

**What we did.** We built the system step by step with small Python scripts, ran each on the user's machine (Windows, RTX 3050 6 GB, 24 GB RAM), and, importantly, **tested every claim with controls and confidence intervals**. That testing changed the direction of the project several times.

**Where we are (headline).**

| # | Finding | Strength |
|---|---------|----------|
| 1 | Early "imagination helps" numbers were an artifact: a trained extra token acts as a **soft prompt**. Controls (shuffled / random / constant tokens) exposed it. | Solid |
| 2 | ScienceQA is a poor test bed: most questions are answerable from text (hints, world knowledge), so swapping the image changes nothing. | Solid (diagnostic script) |
| 3 | The first "spatial" dataset (made by the Cursor agent) looked like a success (oracle 100%) but **leaked the answer** in its images and in answer-position patterns; blank-image accuracy was already 99.2%. | Solid (audit script) |
| 4 | A leak-free dataset (**v2**) shows the image-reading part works: true image beats wrong image by **+8.1 accuracy points (CI 6.4 to 9.9)** and, on tasks where the image is *necessary*, 49.5% vs 35.8% blank. | Solid |
| 5 | On the leak-free data, **text alone already solves most "imagine" families** (about 95%), so imagination has little room to help. Room exists in `grid_walk` (about 81% to 86% text-only) and in `triangle` **out of distribution**. | Solid |
| 6 | **Most interesting:** on out-of-distribution `triangle` problems (bigger numbers), text-only drops to chance (52%) while the true sketch gives 98%. A scale-free picture generalizes where text arithmetic does not. Caveat: the sketch nearly depicts the answer. | Solid as an observation; its meaning is open |
| 7 | **Phase B & C completed (the imagination test):** The Imagination Head successfully learns **example-specific visual semantics** (`own` beats `own_shuffled` by **+1.99% acc [1.21, 2.84]** on `test`, and retrieves the exact ground-truth sketch 60x–80x above chance). However, **single-step imagination from text does not beat the blank-token baseline (`const`)**, and on OOD `triangle`, `own` collapses to chance (**51.7%**) right alongside text-only (**52.0%**). | Solid |
| 8 | **Why OOD single-step imagination failed (The Representation Bottleneck):** The Imagination Head takes the frozen LLM's question hidden state $h$ as input. Because $h$ fails to extrapolate the arithmetic of large OOD numbers, the MLP head cannot construct the scale-free geometry where the arcs meet or miss; instead, it maps OOD numbers back to in-distribution visual prototypes (confirmed by the Phase C retrieval galleries). | Solid |

---

## 2. Glossary of keywords and notation

### 2.1 Core model concepts

| Term | Plain-English meaning |
|------|-----------------------|
| **LLM** (large language model) | A neural network that predicts text, e.g. SmolLM2-135M or Qwen2.5-0.5B (the "backbone"). |
| **Backbone** | The LLM that does the reading and answering. We used SmolLM2-135M first (d_model 576), then Qwen2.5-0.5B (d_model 896). |
| **Hidden state `h_t`** | The LLM's internal vector (size `d_model`) at the last token of reasoning step `t`. It summarizes "what the model has understood so far". |
| **`d_model`** | Width of the LLM's internal vectors (576 for SmolLM2-135M, 896 for Qwen2.5-0.5B). |
| **Reasoning step `t`** | One segment of the chain of thought (for example one sentence "Step 2: ..."). |
| **CoT** (chain-of-thought) | The model writes intermediate reasoning steps before the answer. |
| **Imagination vector `z_t`** | The visual vector produced from `h_t` ("what the model imagines at step t"). Originally 512 numbers; in the advanced track it is a 3×3 grid of 768-number tokens (see "grid"). |
| **Imagination Head** | The small network that turns `h_t` into `z_t`. In the PDF: one linear map `W2`. Ours: a LayerNorm plus a linear map (steps 2 to 8), then a 3-layer MLP of 20.2 M parameters (Phase B). |
| **`W1`, `W2`** | The two weight matrices of the PDF: `W2` makes the imagination, `W1` makes the gate. |
| **Gate `g_t`** | `sigmoid(W1·h_t)` in [0, 1]; high means "show the imagined image", low means "keep it internal". |
| **Feedback loop** | `z_{t-1}` is converted into extra input token(s) and fed into the LLM at step t, so earlier imagination influences later thinking. |
| **Anti-collapse / diversity loss** | A penalty when consecutive imagination vectors are too similar (hinge on cosine similarity). Prevents the "all imagined tokens identical" failure. |
| **Contrastive loss / InfoNCE** | A loss that pulls a vector toward its matching target and pushes it away from other items in the batch ("this imagination must belong to THIS question, not any question"). |
| **Image decoder** | Turns `z_t` into a visible picture. We only used a stand-in: retrieving the closest real image (step 7, Phase C). A real generator (unCLIP / Stable Diffusion) is future work. |
| **Reader** | (advanced track) A small MLP that turns CLIP image features into LLM input embeddings ("visual tokens"), like the projector in LLaVA-style models. It also holds learned "blank" tokens. |
| **Visual tokens** | Vectors inserted into the LLM input that carry image information. |
| **CLIP** | A pretrained model (ViT-B/32) that maps images and text into a shared space; we use its image features as the "picture" representation (512-d global vector, or patch tokens of 768-d). |
| **Patch tokens / grid** | CLIP splits an image into 7×7 patches. "Grid 3" averages them into a 3×3 grid (9 tokens × 768-d); "grid 7" keeps all 49. "Global" = one 512-d vector for the whole image. |
| **LoRA** | A cheap way to fine-tune an LLM by training small add-on matrices instead of all weights (here 2.16 M trainable of 496 M parameters). |
| **bf16 / autocast** | A 16-bit number format used to save GPU memory and time. |
| **Gradient checkpointing** | Recomputes activations during backprop to save memory (slower, but lets a 0.5B model train on 6 GB). |
| **GRPO** | A reinforcement-learning method (Group Relative Policy Optimization) planned for Stage 3 of the PDF; steps 9 and 10 implement a version but were never run. |
| **KL / `kd`** | Distillation loss: make the answer distribution with imagined tokens match the one with the true image. |

### 2.2 Experimental conditions (used in nearly every table)

| Condition | What the model gets as "visual tokens" | Why we test it |
|-----------|----------------------------------------|----------------|
| **oracle** | Tokens from the TRUE image of this question. | Upper bound: "if the model could see the picture". |
| **wrong** | Tokens from ANOTHER question's image. | Tests whether the model uses image *content* (a wrong picture should hurt or at least not help). |
| **const / blank** | Learned "null" tokens (no image information). | Tests what the text alone achieves. (Trained 10% to 15% of the time so it is in-distribution.) |
| **own** | Tokens produced by the model's own Imagination Head from the question text only. | The real test: a "blank-slate student" imagining by itself, with no image. |
| **own_shuffled** | The imagination made for a different question. | Tests whether the imagination is specific to this example. |
| **random_z** | Random unit vectors (step 8b). | Control for "any vector in that slot works". |
| **feedback** (steps 4 to 8d) | `z_t` fed back through the projector. | The original loop. |
| **constant** (steps 4 to 8d) | The same fixed token each step. | Control for the soft-prompt effect. |

### 2.3 Metrics and statistics

| Term | Meaning |
|------|---------|
| **Accuracy (acc)** | Fraction of questions answered correctly (answer letter chosen among valid choices). |
| **CE / NLL** (cross-entropy / negative log-likelihood) | Average "surprise" at the correct answer; lower is better. 0 = perfectly confident and right. |
| **Chance** | Accuracy / CE of random guessing (e.g. 1/4 for four choices; CE = ln(number of choices)). |
| **LM loss** | CE over all text tokens (used in steps 4 to 8d). |
| **Restricted CE** | CE computed only over the valid answer letters (advanced track). |
| **Bootstrap 95% CI** | A confidence interval from resampling the test questions. If it excludes 0, the difference is unlikely to be noise **across questions** (it does not cover run-to-run randomness; every result here is a single training run). |
| **SUPPORTED / OPPOSITE / not distinguishable from 0** | Labels printed by our scripts: CI above 0 / CI below 0 / CI contains 0. |
| **Paired contrast** | A comparison of two conditions on the same questions, e.g. `oracle vs wrong`. |
| **Top-k retrieval** | Fraction of cases where the correct item is among the k most similar. **Median rank** = typical position of the correct item. |
| **Consecutive cosine** | Average cosine similarity between `z_t` and `z_{t-1}`; near 1 means collapse. |
| **Headroom** | Room for improvement: if blank-token accuracy is already about 100% on a task, imagination cannot help there. |
| **`imagination_benefit`** (step 8) | LM loss with constant token minus LM loss with feedback. Positive looked good but was later shown to be mostly a soft-prompt effect. |
| **`visual_info_usable`** | `oracle_wrong − oracle` in loss: can the LLM use image *content* at all? |
| **`own_info_gain`** | `own_shuffled − own` in loss: does the model's own imagination help THIS example? |
| **Recovery** | `own_info_gain / visual_info_usable` (or `(const - own) / (const - oracle)` in Phase B); meaningless when the denominator is tiny. |

### 2.4 Training and data concepts

| Term | Meaning |
|------|---------|
| **Teacher forcing** | During training, feed the true thing (here the true image tokens) instead of the model's own output. At test time the model must use its own. |
| **Scheduled sampling** | Randomly mixing teacher-forced and own outputs during training. |
| **Blank-slate student / "imagine by itself"** | The design principle: at test time no image is given. Images are only a teaching signal. |
| **Soft prompt** | An extra trained input vector that helps simply by being there (like prompt-tuning), not because of what it encodes. Our main false positive. |
| **Answer-first format** | `[question + choices] [visual tokens] "Answer:" → letter`. The loss is only on the answer letter. |
| **Phase A / A0 / B / C** | A0 = caption alignment of the reader; A = teach the reader to use true images; B = train the Imagination Head (reader frozen); C = final test evaluation. |
| **Alignment stage (A0)** | Training the reader on image-caption pairs (COCO) with the LLM frozen, so the LLM learns to "read" visual tokens (as in LLaVA). |
| **Track: `imagine` / `see`** | `imagine`: the question TEXT fully specifies the scene, so a student can imagine it. `see`: the text says nothing about the scene, so the image is necessary. |
| **Family** | One task type inside the spatial datasets (e.g. `triangle`, `grid_walk`). |
| **Split: train / val / test / test_ood** | Training data / monitoring data / final held-out data / harder held-out data (bigger numbers, grids, longer walks). |
| **OOD** (out-of-distribution) | Test cases that differ from the training distribution (here: larger values, longer move sequences). |
| **Leak** | Any way to get the answer without doing the task: answer written in the picture, answer-coded colors, answer always at index 0, choices that encode the answer, etc. |
| **Audit** | `imagine_audit_data.py`: cheap "attacker" tests that look for leaks. |
| **Ablation / control** | A deliberately crippled variant used to check what really causes an effect. |
| **Distractor** | A wrong answer option. |
| **Scale-free picture** | A drawing normalized (to the longest side) so the same figure appears at any numeric scale. |
| **`LIB_VERSION`** | A number in `imagine_adv_lib.py` so other scripts can tell if it is outdated. |
| **Cursor agent** | The AI coding assistant the user also used; it produced the first spatial dataset (v1). |
| **Dual-stack idea (future)** | The user's later idea: two transformer stacks (text and latent image) with bidirectional cross-attention after each self-attention pass. Not yet started. |

---

## 3. How to read the result tables

1. **Accuracy and CE for oracle / wrong / const (and own).** The most important comparison is **oracle vs wrong**: if the model uses the image, a wrong image must be worse than the right one. For imagination, the key comparisons are **own vs const** (does imagining beat doing nothing?) and **own vs own_shuffled** (is the imagination specific to this question?).
2. **Always look at `const` (blank) first.** If blank-token accuracy is near 100%, the task is solvable without the image, so no image-based method can show a benefit.
3. **Per-track and per-family tables** show where an effect lives. Overall averages hide the structure (for example, one family can drive the whole OOD gain).
4. **A CI that contains 0 means "no evidence of a difference"**, not "proven equal". With about 600 to 2000 questions we can detect differences of roughly 1 to 3 accuracy points.
5. **Multiple comparisons:** when a table has many lines, one "SUPPORTED" can appear by chance (we saw this at epoch 2 of the first Phase A run).

---

## 4. The idea and how our experiments differ from the original PDF

```
                 question text
                       |
                       v
        +-------------------------------+
        |  LLM (Qwen2.5-0.5B + LoRA)    |---> hidden state h  (end of question)
        +-------------------------------+           |
                 ^         |                        v
                 |         |              Imagination Head (MLP)
                 |         |                        |
                 |         |                        v
                 |         |           z = 3x3 grid of 768-d tokens ("imagined picture")
                 |         |                        |
            visual tokens  |                        |
                 ^         |                        |
                 +---- Reader (MLP) <---------------+
                           |
                           v
                  "Answer:"  ->  letter A/B/C/D

 TEACHER (training only): true image -> CLIP -> 3x3x768 features -> Reader -> LLM
 TEST: no image anywhere; the model uses only its own imagination ("own").
```

| Aspect | Original PDF | What we actually ran |
|--------|--------------|----------------------|
| Imagination | One 512-d vector per step, `z_t = W2·h_t` | Steps 2 to 8: that (plus LayerNorm). Advanced track: 3×3 grid of 768-d CLIP-patch tokens predicted by a 3-layer MLP. |
| Where imagination is injected | Fed back at every step | Steps 4 to 8d: after every step (1 token, later 4). Advanced track: **one** imagination before the answer (answer-first). The multi-step loop returns later (proposed for `grid_walk`). |
| Gate | `sigmoid(W1·h)`, trained with correctness labels | Trained in step 6 with a proxy label ("question has an image in ScienceQA"); not part of the advanced track yet. |
| Anti-collapse | Contrastive loss `z_t ≠ z_{t-1}` | Implemented (hinge on consecutive cosine) in steps 3 and 8; across-batch InfoNCE in Phase B. Collapse across steps cannot be measured in the single-imagination advanced track. |
| Backbone | e.g. LLaMA | SmolLM2-135M, then Qwen2.5-0.5B (fits a 6 GB GPU) |
| Datasets | COCO, Visual CoT, ScienceQA | COCO captions (yes), ScienceQA (yes, but a poor test), Visual CoT (skipped: about 143 GB images), plus our own spatial datasets |
| GRPO | Stage 3 refinement | Written (steps 9 and 10), **never run** |
| Image decoder | Present | Retrieval stand-in only (Step 7 and Phase C galleries) |

**Principle adopted from the user:** at test time the model starts as a blank slate and must imagine by itself, like a student hearing a teacher's problem. Images are only a training-time teaching signal, analogous to teacher forcing in next-token training.

---

## 5. Timeline of what was done, with every intermediate result

*(Terminal timestamps in parentheses come from the user's logs. Numbers are quoted from the terminal output.)*

### Phase 1: Building the basic modules (steps 1 to 7), SmolLM2-135M

**Step 1, `imagine_step1.py` (2026-10-21 00:30).** Loads the backbone and extracts hidden states.
- Output: `d_model=576`, hidden shape `(1, 28, 576)`, `h_t` norm 49.47, generation continues sensibly ("Step 1: The diameter of the ball is 9 cm. Step 2: ...").
- A harmless print bug (batch shape printed as a tensor) was fixed later.
- Meaning: the backbone works and we can read `h_t`.

**Step 2, `imagine_step2.py`.** Builds the Imagination Head and gate (`z = W2·h`, `g = sigmoid(W1·h)`, with LayerNorm).
- Output: z shape `(1, 18, 512)`, gate shape `(1, 18)`, z norms exactly 1.0, gate values 0.56 to 0.66, 297,153 trainable parameters.
- Meaning: correct shapes; random-init gates sit near 0.5.

**Step 3, `imagine_step3.py`.** Demonstrates the anti-collapse loss on a deliberately collapsed head.
- Output: consecutive cosine **1.000 → -0.286** after 300 iterations (target at or below 0.3).
- Meaning: the loss works mechanically. It went negative because the hinge pushes hard; harmless. A caveat raised then: "z_t ≠ z_{t-1}" alone can be satisfied by random noise, so it needs an anchor (done later).

**Step 4, `imagine_step4.py` (the feedback loop).** Context for step t is `[I_-1, seg_0, I_0, seg_1, ..., I_{t-1}, seg_t]` where `I_t = proj(z_t)`.
- Check 1: LM loss feedback 1.5943 vs constant 1.5904, relative change of final `h` 0.0757 (must be above 0): feedback changes the LLM.
- Check 2 (causality): max change in earlier z when editing the last step: **0.00e+00**; last step 1.24e-02: no information leaks backwards.
- Check 3: gradient norms W2 0.80, proj 0.73, W1 0.0 (the gate is trained separately).
- Meaning: the loop is wired correctly.

**Step 5, `imagine_step5.py` (Stage 1: COCO captions → CLIP image embeddings).**
- Data: the first source (`nlphuji/...`) failed ("Dataset scripts are no longer supported"); the fallback `lmms-lab/COCO-Caption` gave 2000 images in 1:24. A crash on loading CLIP (torch < 2.6 refuses `.bin` weights) was fixed with `use_safetensors=True`.
- Result on 200 held-out images: before training top-1 0.01 / top-5 0.025; **after training top-1 0.34, top-5 0.69, top-10 0.815, median rank 3** (chance 0.005 / 0.025). Validation stopped improving about epoch 20 while training loss kept falling (0.37 at epoch 40): mild overfitting.
- Qualitative: a query "A large passenger jet flying through a gray sky" retrieved other jets; kitchens retrieved kitchens.
- Meaning: a linear head on a small LLM's hidden state does carry visual semantics.

**Step 6, `imagine_step6.py` (gate).** ScienceQA items with an image labeled 1, without labeled 0 (400 + 400).
- Result: validation accuracy **0.956** (train 0.998; before training 0.481).
- But the hand-written probes were wrong in two places: the ball-in-box question got g=0.029 (should be high) and "What is the capital of France?" got g=0.986 (should be low).
- Meaning: the gate learned dataset habits (ScienceQA's "capital" questions come with maps), so "has image" is only a **proxy label**. The user asked to keep the pipeline general rather than patch for probes; the real label ("does this need visual imagination") will need an LLM judge later.

**Step 7, `imagine_step7.py` (stand-in decoder = nearest real image).**
- Gates: ball-box 0.980, bus 0.999, dog 0.944, "broccoli and rice" 0.401 (hidden), "capital of France" 1.000.
- Retrieval similarities 0.20 to 0.28 (typical for CLIP). Bus, dog and food retrieved sensible images; ball-and-box did not (the 2000-image gallery probably has no such photo).
- Meaning: the head maps text into a meaningful visual region; the gate is noisy because of the proxy label.

### Phase 2: Joint training and the first controls (steps 8, 8b, 8c, 8d)

**Step 8, `imagine_step8.py`.** Backbone frozen; trains head, gate and feedback projector on ScienceQA chains-of-thought (480 examples; 400 train / 80 val) with five losses (LM, alignment to CLIP text of each step, image alignment, anti-collapse, gate).
- Before: `lm_feedback 1.6543 | lm_constant 1.9989 | imagination_benefit 0.3447 | consecutive cos 0.861 | align_top1 0.035 | gate_acc 0.775`.
- After 2 epochs (about 8 min each): `lm_feedback 1.3324 | lm_constant 2.0355 | benefit 0.7031 | cos 0.787 | align_top1 0.182 (chance 0.004) | gate_acc 0.943`.
- Looked good: the loss fell and alignment rose 45× chance.

**Step 8b, `imagine_step8b.py` (control: is the imagination informative?).**
- Mean LM loss: feedback **1.3324**, shuffled **1.3451**, random_z **1.5630**, constant **2.0355**.
- Information gain (shuffled minus feedback) only **+0.0127**; soft-prompt effect (constant minus shuffled) **+0.6905**. Consecutive cosine 0.843 (model) vs 0.816 (the gold CLIP step texts themselves), so no collapse.
- **Key finding #1:** nearly all of the "benefit" in step 8 was a trained extra token acting as a soft prompt, not information in `z`. The LLM is text-only, and `z` is computed from the text it already read, so it cannot add anything new by construction.

**Step 8c, `imagine_step8c.py` (teacher forcing with the true image, k=4 tokens, LoRA).** Tested "learn to imagine with the image as teacher; at test time use own imagination".
- Final (60 val image questions): own 1.0844 | own_shuffled 1.0971 | oracle 1.0228 | oracle_wrong 1.0381 | constant 1.1038.
- `visual_info_usable` +0.0153, `own_info_gain` +0.0126 (it was +0.0073, −0.0001, +0.0126 across epochs), "recovery" jumped 3.32, −0.02, 0.82. Blank-slate z_0 retrieves the true image with top-1 0.083 (chance 0.017), cosine to right image 0.685 vs 0.617 to a wrong one.
- Meaning: numbers had the right sign but were tiny and unstable on 60 questions. "Recovery" is meaningless when its denominator is about 0.002 to 0.015.

**Step 8d, `imagine_step8d.py` (150 fresh questions, bootstrap CIs, plus an "answer-first" readout).**
- CoT setting: own 1.1391, shuffled 1.1440, oracle 1.0774, wrong 1.0784, constant 1.1600. `own vs blank` +0.0208 CI [−0.0042, +0.0436]; `own is example-specific` +0.0049 CI [−0.0250, +0.0347]; `image content usable` **+0.0010 CI [−0.0004, +0.0025]** (so 8c's +0.015 was noise).
- Answer-first (chance CE 1.004): own 1.2492, oracle 1.2587, wrong 1.2597, constant 1.3395. `own vs blank` +0.0903 SUPPORTED, but oracle equals own and wrong equals oracle.
- **Key finding #2:** the LLM did not read image content at all; every image-like token worked equally. The test bed itself (gold reasoning steps in context, text-only LLM) could not show a visual benefit.

### Phase 3: The "advanced track" on ScienceQA (Qwen2.5-0.5B, answer-first)

**Design change.** Use a bigger model (Qwen2.5-0.5B + LoRA, bf16, gradient checkpointing), official ScienceQA splits, **image questions only**, answer-first format, and a **reader** (CLIP features → LLM tokens). Programs: `imagine_adv_lib.py` plus `imagine_step8_advanced_a/b/c.py`. Success test for Phase A: **oracle must beat wrong with a confidence interval above 0.**

**Phase A runs on ScienceQA (4000 train, 600 val):**

| Run | Setting | oracle / wrong / const accuracy (final) | oracle vs wrong |
|-----|---------|------------------------------------------|-----------------|
| 1 (2026-10-21 15:57) | global CLIP vector | 71.8 / 72.3 / 73.2 | not distinguishable |
| 2 (16:28) | grid 3×3 | 70.3 / 70.3 / 70.0 | OPPOSITE by a hair (CE −0.0081) |
| 3 (16:45) | grid 3×3 + wrong-image margin loss (`--w_contrast 1.0`) | 71.5 / 71.8 / 71.7 | OPPOSITE (CE −0.0088) |

Observations: accuracy rose from about 52% to about 72% under **every** condition, including blank, so the model learned ScienceQA's text patterns. CE got worse after epoch 2 (0.618 → 0.755) while accuracy stayed flat (overfitting). A single "SUPPORTED" at epoch 2 of run 1 (+0.0039) vanished by epoch 3: chance.

**A0, caption alignment, `imagine_step8_advanced_a0.py` (about 17:20).** Reader trained on 20,000 COCO images (download 24 min, CLIP features 5:47) with the LLM frozen.
- Held-out caption CE on 500 images: before true 3.817 / wrong 3.818 / blank 3.694. After epoch 2: **true 2.447 / wrong 3.476 / blank 2.939**; gap vs wrong **+1.029 [0.988, 1.070]**, vs blank +0.492 [0.469, 0.515].
- **Key finding #3:** the visual path works: a frozen Qwen can read CLIP grid tokens after only about 8 minutes of alignment.

**Phase A again on ScienceQA with the aligned reader:** final oracle 69.5 / wrong 69.2 / const 68.5, still not distinguishable (before training, any image token gave +7.7 points over blank, but not the right image over a wrong one).
- **Key finding #4:** the failure is not the visual path; **the dataset gives the image nothing to add** (see Phase 4).

### Phase 4: Diagnosing ScienceQA (`imagine_diagnose_data.py`, written by Cursor)
- Validation set (n=600): 65% have a hint text, 50% have a long hint (≥ 80 characters, often restating the figure), **14% contain the gold answer string** in the question or hint, 26% look like knowledge or meta questions.
- Train set (n=4000): 63% with hint, 48% long hint, 12% gold string present, 25% knowledge questions.
- Interpretation: swapping the figure rarely changes the answer, so oracle ≈ wrong. A dataset failure, not evidence against imagination.

### Phase 5: The first spatial dataset (v1, built by the Cursor agent), which "passed" for the wrong reason

`imagine_build_spatial_cot.py` produced 80,000 / 4,000 / 4,000 items in 8 families (`fit_ball_box`, `fit_rect`, `triangle`, `compare_len`, `clock`, `spatial_imagine`, `count_see`, `mark_see`).

**Phase A on v1** (`--data spatial`, 20,000 train, 2,000 val, contrast loss on; about 40 to 46 min per epoch):
- Before: oracle 33.7 / wrong 34.4 / const 36.8.
- Epoch 1: **96.2 / 49.9 / 76.9**; epoch 2: 99.6 / 55.7 / 88.5; epoch 3: 99.9 / 57.7 / 98.6; **epoch 4: 100.0 / 58.3 / 99.2**.
- Printed verdict "PHASE A PASSED", but **blank tokens reached 99.2%**, and "image helps vs blank" was only +0.8 points. Text alone solved the task; and a wrong image was *worse than blank* (58%), meaning the model followed a cue in the picture.

**Audit (`imagine_audit_data.py`, written by Claude) on v1 val (n=4000): 8 of 8 families flagged.**

| Family | Leak found |
|--------|-----------|
| `fit_ball_box`, `fit_rect`, `triangle` | A single color count (RED or GREEN pixels) predicts the answer 100% from the image alone: the sketches were green for "fits" and red otherwise, and had "fits / does not fit / valid triangle" written on them. |
| `spatial_imagine` | The correct object was ringed in the picture; answer index 0 correct 36% (chance 26%). |
| `mark_see` | **Answer index 0 is correct 100%** (the correct option was always listed first). |
| `count_see` | Options were always gold−1..gold+2, so the choice list encodes the answer (a lookup solves 100%; rank bias 60%). |
| `clock` | Question and choice-set lookups solve 100%. |
| `compare_len` | Question lookup solves 100%. |
| `triangle` | Answer index 0 correct 71% (chance 50%). |

- **Key finding #5:** the "oracle 100%" was the model reading an answer written into the picture, and the "see" track was text-solvable. The result was uninformative.

### Phase 6: The leak-free dataset v2 (`imagine_build_spatial_v2.py`, written by Claude)

Design rules: pictures show the **scene or state, never the verdict**; yes/no labels balanced by construction; choices shuffled so the answer position is uniform; `imagine` track = text fully specifies the scene; `see` track = text says nothing about the scene; a `test_ood` split with harder parameters. Built 40,000 / 3,000 / 3,000 / 2,000 (OOD) items.

Self-audit: **0 of 8 families flagged**. Notes (not leaks): to-scale sketches still encode geometry (for example total arc length predicts `triangle` 99%, orange pixels predict `fit_ball_box` 78%).

**Phase A on v2** (`--data spatial2 --vis grid --grid 3`, 20,000 train / 2,000 val, 4 epochs, about 18 min/epoch, 2026-10-22):

| Epoch | oracle | wrong | const |
|-------|--------|-------|-------|
| before | 38.1 | 37.8 | 36.8 |
| 1 | 74.4 | 66.0 | 72.2 |
| 2 | 80.4 | 72.2 | 76.3 |
| 3 | 81.2 | 73.0 | 76.0 |
| **4 (final)** | **81.8** | **73.8** | **76.9** |

Final contrasts: `oracle vs wrong` **+8.10 points [6.40, 9.85], SUPPORTED**; `oracle vs blank` +4.95 points [3.45, 6.40], SUPPORTED. CE: oracle 0.3659, wrong 0.8544, const 0.4519.

By track and family (final, accuracy %):

| Group | oracle | wrong | const | Reading |
|-------|--------|-------|-------|---------|
| imagine track (n=1386) | 96.2 | 91.3 | 95.1 | text alone is at 95%: little headroom |
| see track (n=614) | 49.5 | 34.2 | 35.8 | image necessary and used (+13.7 over blank) |
| `count_see` | 63.6 | 21.3 | 20.9 | learned (35 → 61 → 63 → 64 over epochs); not exact |
| `mark_see` | 30.9 | 32.7 | 34.6 | chance: ring not readable from a 3×3 grid |
| `rel_see` | 47.9 | 49.8 | 53.5 | chance: left-of / above not readable |
| `fit_ball_box` | 100 | 95.8 | 100 | text solves it |
| `fit_rect` | 100 | 95.0 | 100 | text solves it |
| `spatial_grid` | 100 | 100 | 100 | text solves it |
| `triangle` | 100 | 78.9 | 93.2 | picture helps (+6.8); wrong picture hurts |
| `grid_walk` | 86.2 | 84.9 | 85.6 | the starting board adds nothing; text-only has headroom |

- **Key finding #6:** the reader really uses image content when the image is necessary. A wrong image hurts on several families (it misleads), which means wrongly imagined tokens could also hurt.

**Evaluation only, on the OOD split (`imagine_step8_advanced_a_eval.py`, 2026-10-23 23:59, grid-3 model, 2000 items):**

| Family | oracle | wrong | const |
|--------|--------|-------|-------|
| all (imagine track) | 93.7 | 82.8 | 85.3 |
| `fit_ball_box` | 99.2 | 92.7 | 97.6 |
| `fit_rect` | 99.3 | 91.6 | 99.3 |
| `spatial_grid` | 99.0 | 99.7 | 98.6 |
| `grid_walk` | 77.5 | 76.0 | 75.8 |
| **`triangle`** | **98.2** | **52.6** | **52.0** |

Overall `oracle vs blank` +8.40 points [7.20, 9.60]; `oracle vs wrong` +10.85 [9.45, 12.30].
- **Key finding #7 (most interesting):** with bigger numbers (sides 16 to 40), the text-only path on `triangle` collapses to chance (52%), while the true sketch gives 98%. The sketch is normalized to the longest side (scale-free), so it reads the same at any scale, while text arithmetic does not extrapolate. **Caveat:** the sketch comes close to depicting the answer (arcs meet or not), so the real question is whether a model can *produce* that picture features from the numbers alone.

**Phase A with grid 7 (49 tokens), 2026-10-24 00:02: inconclusive.** Final oracle 76.2 / wrong 75.6 / const 76.1; `count_see` 25% (chance). The log lacks the line "reader initialised from the caption-alignment stage A0": the A0 checkpoint is saved per grid size and none existed for grid 7, so this run started with an unaligned reader. It tested nothing about finer tokens (a mistake in our run list: A0 for grid 7 was a missing prerequisite). To redo: run `imagine_step8_advanced_a0.py --vis grid --grid 7` first.

### Phase 7: Phase B — Training the Imagination Head (Completed, 2026-10-24 11:30 to ~12:25)

`imagine_step8_advanced_b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3`: trained **only the Imagination Head** (20.2 M parameters) with the reader and LLM frozen. It takes the question text's final hidden state `h` (dimension 896) and predicts a $3\times 3\times 768$ grid of CLIP features. The true image acts only as a teacher through four losses:
1. `feat`: cosine distance + $0.1 \times \text{MSE}$ against layer-normalized true CLIP patch tokens.
2. `nce`: symmetric batch InfoNCE (after centering by the batch mean) so imagined tokens stay specific to their question.
3. `ans`: cross-entropy on the answer letter when feeding `reader(imag)` into the frozen LLM.
4. `kd`: KL divergence matching the teacher's answer probability distribution (`lg_or` vs `lg_own`).

Only `imagine` track items were used (filtered 20,000 train down to **14,017** items, and 2,000 val down to **1,386** items; 876 steps/epoch at ~17 minutes/epoch).

**Progression across training on `val` ($n=1386$, chance acc 37.6%, CE 1.038):**

| Stage | `oracle` (acc / CE) | `own` (acc / CE) | `own_shuffled` (acc / CE) | `wrong` (acc / CE) | `const` (acc / CE) | Retrieval Top-1 / Top-5 (chance 0.001 / 0.004) |
|-------|---------------------|------------------|---------------------------|--------------------|--------------------|------------------------------------------------|
| **Before (random head)** | 96.2% / 0.1332 | 94.6% / 0.1849 | 95.0% / 0.1764 | 90.7% / 0.5044 | 95.1% / 0.1561 | 0.000 / 0.001 |
| **Epoch 2** | 96.2% / 0.1332 | 95.2% / 0.1612 | 94.2% / 0.2059 | 90.7% / 0.5044 | 95.1% / 0.1561 | 0.045 / 0.192 |
| **Epoch 3 (Final)** | **96.2% / 0.1332** | **95.2% / 0.1592** | **92.8% / 0.3040** | **90.7% / 0.5044** | **95.1% / 0.1561** | **0.069 / 0.235** |

*(Training loss logs at Epoch 3 end: `feat=0.066, nce=0.365, ans=0.091, kd=0.003, acc=0.969`.)*

**Per-family breakdown on `val` at Final (Epoch 3, accuracy %):**

| Family | `n` | `oracle` | `own` | `own_shuffled` | `wrong` | `const` |
|--------|-----|----------|-------|----------------|---------|---------|
| `fit_ball_box` | 306 | 100.0 | 100.0 | 96.1 | 91.2 | 100.0 |
| `fit_rect` | 200 | 100.0 | 100.0 | 98.5 | 95.5 | 100.0 |
| `grid_walk` | 383 | 86.2 | 86.2 | 85.9 | 85.9 | 85.6 |
| `spatial_grid` | 307 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `triangle` | 190 | 100.0 | **93.2** | **83.7** | 79.5 | **93.2** |

**Key paired contrasts at Final (Epoch 3 on `val`):**
- `own imagination vs blank (const - own)`: CE $-0.0032$ $[-0.0116, +0.0048]$ | acc $+0.14$ $[-0.29, +0.58]$ pts $\to$ **not distinguishable from 0**.
- `own is example-specific (own_shuffled - own)`: CE $+0.1448$ $[+0.0885, +0.2066]$ | acc **$+2.45$ $[+1.52, +3.39]$ pts $\to$ SUPPORTED**.
- `own vs a random real image (wrong - own)`: CE $+0.3452$ $[+0.2535, +0.4413]$ | acc **$+4.55$ $[+3.39, +5.77]$ pts $\to$ SUPPORTED**.
- `headroom: true image vs own (own - oracle)`: CE $+0.0260$ $[+0.0125, +0.0417]$ | acc $+0.94$ $[+0.51, +1.44]$ pts $\to$ **SUPPORTED**.

**What happened during Phase B:**
1. **The head genuinely learned to imagine specific scenes:** Before training, `own` (94.6%) and `own_shuffled` (95.0%) were identical and worse than `const` (95.1%). By Epoch 3, `own` rose to 95.2% (matching `const`), while `own_shuffled` dropped to 92.8% (and from 89.5% down to 83.7% on `triangle`). This proves the Imagination Head stopped emitting generic noise and started generating **question-specific visual tokens** that actively mislead the reader if swapped to a different question!
2. **Strong visual retrieval:** Retrieval of the exact ground-truth diagram among 1,386 validation sketches jumped from $0.000$ to **6.9% Top-1** ($69\times$ chance) and **23.5% Top-5** ($59\times$ chance)—remarkable given that dozens of synthetic sketches share nearly identical geometric parameters.
3. **No net gain over blank tokens in-distribution:** On `triangle` (the main family with in-distribution headroom between `const` 93.2% and `oracle` 100.0%), `own` reached exactly **93.2%**, recovering none of the $+6.8\%$ visual benefit over `const`.

---

### Phase 8: Phase C — Final Held-Out Evaluation on `test` and `test_ood` (2026-10-24 12:36 to 12:41)

We ran `imagine_step8_advanced_c.py` twice on held-out splits never seen during training or model selection: once on the in-distribution `test` split (filtered to **1,409** `imagine` items) and once on the out-of-distribution `test_ood` split (**2,000** `imagine` items with larger numbers, bigger grids, and longer walks).

#### 8A. In-Distribution Held-Out `test` Split ($n=1409$; chance acc 37.4%, CE 1.041)

| Condition | Accuracy (%) | Cross-Entropy (CE) |
|-----------|--------------|--------------------|
| `oracle` (true image) | **97.3%** | **0.1001** |
| `own` (student's imagination) | **96.7%** | **0.1247** |
| `const` (blank tokens) | **96.5%** | **0.1248** |
| `own_shuffled` (another Q's imagination) | 94.7% | 0.2079 |
| `wrong` (another Q's real image) | 91.7% | 0.4710 |

**Paired bootstrap contrasts (`test`, 95% CI):**
- `own imagination vs blank (const - own)`: CE $+0.0001$ $[-0.0131, +0.0123]$ | acc **$+0.21$ $[-0.14, +0.57]$ pts $\to$ not distinguishable from 0**
- `own is example-specific (own_shuffled - own)`: CE $+0.0832$ $[+0.0417, +0.1291]$ | acc **$+1.99$ $[+1.21, +2.84]$ pts $\to$ SUPPORTED**
- `own vs a random real image (wrong - own)`: CE $+0.3463$ $[+0.2517, +0.4482]$ | acc **$+4.97$ $[+3.83, +6.17]$ pts $\to$ SUPPORTED**
- `headroom: true image vs own (own - oracle)`: CE $+0.0246$ $[+0.0098, +0.0434]$ | acc **$+0.64$ $[+0.28, +1.14]$ pts $\to$ SUPPORTED**
- `[reference] image content usable (wrong - oracle)`: CE $+0.3709$ $[+0.2749, +0.4737]$ | acc **$+5.61$ $[+4.40, +6.88]$ pts $\to$ SUPPORTED**
- `[reference] true image vs blank (const - oracle)`: CE $+0.0247$ $[+0.0096, +0.0429]$ | acc **$+0.85$ $[+0.35, +1.42]$ pts $\to$ SUPPORTED**

**Per-family accuracy (`test`, %):**

| Family | `n` | `oracle` | `own` | `own_shuffled` | `wrong` | `const` |
|--------|-----|----------|-------|----------------|---------|---------|
| `fit_ball_box` | 301 | 100.0 | 100.0 | 98.7 | 91.7 | 100.0 |
| `fit_rect` | 197 | 100.0 | 100.0 | 99.0 | 94.9 | 100.0 |
| `grid_walk` | 382 | 90.1 | 90.1 | 88.5 | 87.4 | 90.1 |
| `spatial_grid` | 326 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `triangle` | 203 | 100.0 | **95.6** | 87.7 | 83.3 | **94.1** |

- **Image retrieval (`test`):** **Top-1 = 0.060 (6.0%), Top-5 = 0.230 (23.0%)** over 1,409 test images (chance $0.001 / 0.004$).
- **Takeaway on `test`:** Confirmed the validation findings on fresh data. `own` is slightly ahead of `const` on `triangle` (95.6% vs 94.1%, $+1.5\%$), but across the full test set the $+0.21$ pt difference is within noise. Meanwhile, `own` strongly beats `own_shuffled` ($+1.99$ pts, SUPPORTED).

---

#### 8B. Out-of-Distribution Held-Out `test_ood` Split ($n=2000$; chance acc 40.3%, CE 0.963)

| Condition | Accuracy (%) | Cross-Entropy (CE) |
|-----------|--------------|--------------------|
| `oracle` (true image) | **93.7%** | **0.2097** |
| `own` (student's imagination) | **85.9%** | **0.9331** |
| `const` (blank tokens) | **85.3%** | **0.7806** |
| `own_shuffled` (another Q's imagination) | 85.1% | 0.9149 |
| `wrong` (another Q's real image) | 82.8% | 1.1570 |

**Paired bootstrap contrasts (`test_ood`, 95% CI):**
- `own imagination vs blank (const - own)`: CE **$-0.1525$ $[-0.2004, -0.1042]$** | acc **$+0.60$ $[+0.15, +1.05]$ pts $\to$ OPPOSITE** *(Note: the script's verdict is driven by CE, where `own` is worse/more overconfident on errors, even though `own` has a slightly positive $+0.60$ pt accuracy edge over `const`!)*
- `own is example-specific (own_shuffled - own)`: CE $-0.0182$ $[-0.0881, +0.0560]$ | acc **$+0.80$ $[+0.15, +1.55]$ pts $\to$ not distinguishable from 0** *(in CE; positive in accuracy)*
- `own vs a random real image (wrong - own)`: CE $+0.2238$ $[+0.1036, +0.3479]$ | acc **$+3.05$ $[+1.90, +4.20]$ pts $\to$ SUPPORTED**
- `headroom: true image vs own (own - oracle)`: CE $+0.7234$ $[+0.6119, +0.8361]$ | acc **$+7.80$ $[+6.60, +9.00]$ pts $\to$ SUPPORTED**
- `[reference] image content usable (wrong - oracle)`: CE $+0.9472$ $[+0.8162, +1.0824]$ | acc **$+10.85$ $[+9.45, +12.30]$ pts $\to$ SUPPORTED**
- `[reference] true image vs blank (const - oracle)`: CE $+0.5709$ $[+0.4805, +0.6592]$ | acc **$+8.40$ $[+7.20, +9.60]$ pts $\to$ SUPPORTED**

**Per-family accuracy (`test_ood`, %):**

| Family | `n` | `oracle` | `own` | `own_shuffled` | `wrong` | `const` |
|--------|-----|----------|-------|----------------|---------|---------|
| `fit_ball_box` | 490 | 99.2 | **99.2** | 96.9 | 92.7 | 97.6 |
| `fit_rect` | 404 | 99.3 | 99.0 | 97.0 | 91.6 | 99.3 |
| `grid_walk` | 488 | 77.5 | **76.8** | 76.4 | 76.0 | 75.8 |
| `spatial_grid` | 291 | 99.0 | 99.0 | 99.3 | 99.7 | 98.6 |
| **`triangle`** | 327 | **98.2** | **51.7** | **52.9** | **52.6** | **52.0** |

- **Image retrieval (`test_ood`):** **Top-1 = 0.012 (1.2%), Top-5 = 0.047 (4.7%)** over 2,000 OOD test images (chance $0.001 / 0.003$; still $12\times$ to $16\times$ above chance, though lower than in-distribution).
- **Takeaway on `test_ood` (The Crucial Result):**
  1. **On OOD `triangle`, `own` completely fails to recover the oracle sketch's benefit:** `oracle` gets **98.2%**, while `own` gets **51.7%**—indistinguishable from `const` (**52.0%**, random chance). Because `own` is trained to match the confident teacher (`kd` loss), its errors on OOD `triangle` carry high confidence, inflating its average CE from $0.7806$ (`const`) to $0.9331$ (`own`) and triggering the `OPPOSITE` CE verdict.
  2. **Small accuracy gains on other OOD families:** On `fit_ball_box` ($+1.6\%$ over `const`, matching `oracle` at 99.2%) and `grid_walk` ($+1.0\%$ over `const`), `own` edges out `const`, giving an overall accuracy gain of **$+0.60$ pts $[+0.15, +1.05]$** on `test_ood`.

---

#### 8C. Qualitative Analysis of the Phase C Imagination Galleries (`imagine_out/*.png`)

Inspecting the saved galleries (`adv_c_qwen2.5-0.5b_grid3_spatial2_test_imagination.png` and `..._test_ood_imagination.png`) reveals *precisely* what the Imagination Head learned and why it struggles out of distribution:

1. **In-Distribution Gallery (`test`):**
   - **Exact Family & Layout Matching:** Every single query retrieves top-3 images from the **exact same task family** (`3x3 grid` retrieves `3x3 grid`, `ball in box` retrieves `ball in box`, `4x4 grid_walk` retrieves `4x4/5x5/6x6 grid_walk`).
   - **High-Precision Parameter Matching (Row 2):** For the query *"Can a ball of diameter 10 cm fit inside a box that is 12 cm wide and 8 cm tall?"*, **Imagined #1 retrieves the exact true image (`d=10`, `12 cm` $\times$ `8`)!** Imagined #2 (`d=12`, `14 cm` $\times$ `8`) and Imagined #3 (`d=11`, `13 cm` $\times$ `8`) preserve the exact aspect ratio and the "too tall by 2–4 cm" visual state where the orange circle protrudes above and below the blue box.
   - **Coarse Object/Position Matching:** On `spatial_grid` (Rows 1, 3, 6) and `grid_walk` (Rows 4, 5), the imagined tokens retrieve boards with similar object densities and colors (e.g., Row 5 `4x4 grid_walk` retrieves `4x4` boards with the blue agent circle at row 3), though not always the exact wall coordinates.

2. **Out-of-Distribution Gallery (`test_ood`):**
   - **Why OOD Imagination Fails — The In-Distribution Projection Bias:** When given OOD numbers (which are larger than anything seen in training), the MLP head projects the LLM's hidden state into familiar *in-distribution* visual feature patterns rather than computing exact OOD proportions:
     - **Row 4 (`fit_ball_box`):** Query asks about a ball of `d=24 cm` in a box `26 x 31 cm` (Answer: **Yes**, it fits comfortably). The true sketch shows the circle inside the box. However, the head retrieves sketches with `d=19` in `31 x 23`, `d=17` in `34 x 23`, and `d=15` in `33 x 21`. Because the MLP hasn't learned to extrapolate the ratio $24 < \min(26, 31)$, it retrieves somewhat generic medium-ball-in-wide-box features.
     - **Row 6 (`fit_ball_box`):** Query has `d=19 cm` in `29 x 23 cm` (Answer: **Yes**, $19 < 23$, circle is inside the box). Yet Imagined #1 retrieves `d=30` in `37 x 31`, and Imagined #2/`#3` (`d=20` in `32 x 22`, `d=25` in `36 x 27`) show the circle *overlapping/protruding* from the vertical edges!
     - **Rows 1 & 3 (`fit_rect`):** Query `15 x 26` inside `24 x 38` (vertical elongated rectangle) retrieves `13 x 17` in `20 x 20` and `19 x 17` in `21 x 25` (nearly square boxes).
   - **Fundamental insight:** A feedforward MLP (`VisualImaginationHead`) reading a frozen LLM's last-token hidden state $h$ cannot magically perform out-of-distribution arithmetic that $h$ itself failed to compute. For an "imagined sketch" to generalize like a real scale-free sketch, the model must construct the sketch **compositionally / iteratively** (e.g., drawing or updating state step-by-step), rather than regressing the entire finished diagram in a single shot from $h$.

---

## 6. Programs inventory

### 6.1 Shared modules (imported by other scripts, not run directly)

| File | Role |
|------|------|
| `imagine_data.py` | Banners and seeds, COCO and ScienceQA download and caching, CLIP loading and embedding (uses safetensors), `get_spatial_cot`. |
| `imagine_adv_lib.py` | Library of the advanced track: ScienceQA/spatial loaders (self-contained from LIB_VERSION 4), CLIP grid features, `VisSet`, `VisualReader`, `VisualImaginationHead`, answer-first forward pass, evaluation conditions, bootstrap statistics, per-group breakdown tables. |

### 6.2 Original track (SmolLM2-135M)

| File | What it does | Status |
|------|--------------|--------|
| `imagine_step1.py` | Load backbone, extract `h_t`, helper for batched hidden states. | Run |
| `imagine_step2.py` | Imagination Head + gate classes. | Run |
| `imagine_step3.py` | Anti-collapse loss demo. | Run |
| `imagine_step4.py` | `FeedbackChain` (feedback loop, `z_swap` and `oracle` controls, k tokens). | Run |
| `imagine_step5.py` | Stage 1: COCO captions → CLIP embeddings (InfoNCE). | Run |
| `imagine_step6.py` | Gate training (proxy label). | Run |
| `imagine_step7.py` | Retrieval "decoder" montage. | Run |
| `imagine_step8.py` | Joint training with five losses, LLM frozen. | Run |
| `imagine_step8b.py` | Control experiment (feedback / shuffled / random / constant). | Run |
| `imagine_step8c.py` | Teacher forcing with the true image, LoRA, k=4. | Run |
| `imagine_step8d.py` | Fresh-data evaluation with bootstrap CIs and answer-first readout. | Run |
| `imagine_step9.py` | LoRA joint fine-tuning. | Written, **never run** |
| `imagine_step10.py` | GRPO with accuracy + format rewards. | Written, **never run** |

### 6.3 Advanced track (Qwen2.5-0.5B, answer-first)

| File | What it does | Status |
|------|--------------|--------|
| `imagine_step8_advanced_a0.py` | Alignment: train the reader on COCO captions (LLM frozen); built-in true vs wrong vs blank check. | Run (grid 3 only) |
| `imagine_step8_advanced_a.py` | Phase A: train reader + LoRA on the answer so true images beat wrong ones; optional `--w_contrast`; `--data scienceqa/spatial/spatial2`. | Run 7 times |
| `imagine_step8_advanced_a_eval.py` | Eval-only: oracle / wrong / blank per family on any split (e.g. `test_ood`). | Run once |
| `imagine_step8_advanced_b.py` | Phase B: train only the Imagination Head from question text; `--track imagine`. | **Run & Completed** |
| `imagine_step8_advanced_c.py` | Phase C: final test, per-group table, gallery of what is imagined; `--split test / test_ood`. | **Run & Completed (`test` and `test_ood`)** |

### 6.4 Dataset and tooling scripts

| File | Author | What it does |
|------|--------|--------------|
| `imagine_diagnose_data.py` | Cursor | Shows how text-solvable ScienceQA is (hint length, answer leakage, etc.). |
| `imagine_build_spatial_cot.py` | Cursor | Builds the v1 spatial dataset (leaky). |
| `imagine_fetch_public.py` | Cursor | Optional extra COCO captions, Visual-CoT annotations (about 345 MB) or images (about 143 GB). Not used so far. |
| `imagine_audit_data.py` | Claude | Leak audit (index bias, numeric-rank bias, question/choice lookup, one-feature image attacker). |
| `imagine_build_spatial_v2.py` | Claude | Builds the leak-free v2 dataset (8 families, OOD split, optional step images) and self-audits. |
| `imagine_check_env.py` | Claude | Checks that your scripts and shared modules come from the same version. |

### 6.5 Folders

| Folder | Contents |
|--------|----------|
| `imagine_cache/` | Downloaded and generated data: COCO images, ScienceQA images, CLIP feature caches (`adv_feats_*`), `spatial_cot/` (v1), `spatial_cot_v2/` (v2). |
| `imagine_ckpt/` | Checkpoints per experiment: `head_step5/6.pt`, `chain_step8*.pt`, `adv_a0_*`, `adv_a_*` (reader + LoRA), `adv_b_qwen2.5-0.5b_grid3_spatial2_imagine/head.pt` (trained imagination head). |
| `imagine_out/` | Pictures: `step7_retrieval.png`, `adv_c_qwen2.5-0.5b_grid3_spatial2_test_imagination.png`, and `adv_c_qwen2.5-0.5b_grid3_spatial2_test_ood_imagination.png`. |
| `old_progs/` | Older versions of scripts, with suffix `old`, `old2`, ... (see Section 10). |

---

## 7. Datasets used or built

| Dataset | Source | Size we used | Purpose | Verdict |
|---------|--------|--------------|---------|---------|
| COCO captions | HF `lmms-lab/COCO-Caption` (val) | 2,000 (step 5), 20,000 (A0) | Alignment of text/visual features and of the reader | Works well |
| ScienceQA | HF `derek-thomas/ScienceQA` | 800 to 4,000 items | Stage-3 stand-in; gate proxy labels | **Poor test:** mostly text-solvable |
| Visual CoT | HF `deepcs233/Visual-CoT` (about 143 GB of images) | Not used | PDF Stage 2 | Skipped |
| Spatial v1 | Cursor agent | 80,000 / 4,000 / 4,000 | Spatial reasoning test | **Leaky; do not use** |
| **Spatial v2** | `imagine_build_spatial_v2.py` | 40,000 / 3,000 / 3,000 / 2,000 OOD | Leak-free test of imagination | Audit clean; **current dataset** |
| A-OKVQA (photo QA) | HF `HuggingFaceM4/A-OKVQA` (multiple-choice, COCO photos) | Not used | Planned "photo VQA" option | Loader not written yet |

**Spatial v2 families**

| Family | Track | Text specifies | The picture shows | Why it exists |
|--------|-------|----------------|-------------------|---------------|
| `fit_ball_box` | imagine | box width/height and ball diameter | the box and a ball to scale (no verdict) | the PDF's own example |
| `fit_rect` | imagine | outer and inner rectangle sizes | both rectangles to scale | packing |
| `triangle` | imagine | three stick lengths | two arcs from the ends of the longest side; they meet or not | triangle inequality |
| `spatial_grid` | imagine | objects listed with row/column | the grid with the objects | spatial lookup |
| `grid_walk` | imagine | grid size, walls, start, move list | the starting board (optionally one image per move with `--step_images`) | multi-step state tracking |
| `count_see` | see | nothing about the scene | scattered shapes | needs counting from the image |
| `mark_see` | see | nothing about the scene | a 3×3 grid with one ringed object | needs reading the ring |
| `rel_see` | see | nothing about the scene | two objects | needs left-of / above |

Items have fields: `question`, `choices`, `answer` (index), `family`, `track`, `image_path`, `steps`, `spec`.

---

## 8. Consolidated results

### 8.1 Original track (SmolLM2-135M, ScienceQA CoT)

| Experiment | Key numbers | Takeaway |
|------------|-------------|----------|
| Step 5 | val top-1 0.34, top-5 0.69 (chance 0.005 / 0.025) | text hidden state carries visual semantics |
| Step 6 | gate val acc 0.956, but probes wrong | proxy label is imperfect |
| Step 8 | LM loss feedback 1.65 → 1.33; benefit 0.34 → 0.70 | looked promising |
| Step 8b | shuffled 1.3451 vs feedback 1.3324; constant 2.0355 | benefit is a soft-prompt effect |
| Step 8c | own_info_gain +0.0126 (unstable over epochs) | sign right, magnitude tiny |
| Step 8d | image content usable +0.0010 [−0.0004, +0.0025] | the LLM did not read the image |

### 8.2 Advanced track Phase A (Qwen2.5-0.5B, reader training, accuracy in %, oracle / wrong / const)

| Dataset and setting | before training | final | oracle vs wrong |
|---------------------|-----------------|-------|-----------------|
| ScienceQA, global | 51.8 / 52.8 / 53.7 | 71.8 / 72.3 / 73.2 | not distinguishable |
| ScienceQA, grid 3 | 53.0 / 51.3 / 54.5 | 70.3 / 70.3 / 70.0 | slightly OPPOSITE |
| ScienceQA, grid 3 + contrast | 53.0 / 51.3 / 54.5 | 71.5 / 71.8 / 71.7 | slightly OPPOSITE |
| ScienceQA, grid 3 after A0 | 50.8 / 52.8 / 45.2 | 69.5 / 69.2 / 68.5 | not distinguishable |
| Spatial v1 (leaky), grid 3 + contrast | 33.7 / 34.4 / 36.8 | **100.0 / 58.3 / 99.2** | SUPPORTED, but via leaks |
| **Spatial v2, grid 3** | 38.1 / 37.8 / 36.8 | **81.8 / 73.8 / 76.9** | **+8.10 pts [6.40, 9.85]** |
| Spatial v2, grid 7 (no A0) | 41.5 / 42.8 / 41.5 | 76.2 / 75.6 / 76.1 | not distinguishable (confounded) |
| Spatial v2 `test_ood`, grid 3 (eval only) | n/a | 93.7 / 82.8 / 85.3 | **+10.85 pts [9.45, 12.30]** |

### 8.3 Advanced track Phase B & C (Qwen2.5-0.5B, Imagination Head on Spatial v2 `imagine` track, grid 3)

| Split (`n`) | `oracle` (acc / CE) | `own` (acc / CE) | `own_shuffled` (acc / CE) | `wrong` (acc / CE) | `const` (acc / CE) | `own vs const` (acc / CE) | `own vs shuffled` (acc / CE) | Retrieval Top-1 / Top-5 |
|-------------|---------------------|------------------|---------------------------|--------------------|--------------------|---------------------------|------------------------------|-------------------------|
| **Val — Before** (1386) | 96.2% / 0.1332 | 94.6% / 0.1849 | 95.0% / 0.1764 | 90.7% / 0.5044 | 95.1% / 0.1561 | $-0.51$ pts / $-0.0288$ (OPPOSITE) | $-0.43$ pts / $-0.0084$ (null) | 0.000 / 0.001 |
| **Val — Final Ep 3** (1386) | 96.2% / 0.1332 | 95.2% / 0.1592 | 92.8% / 0.3040 | 90.7% / 0.5044 | 95.1% / 0.1561 | $+0.14$ pts / $-0.0032$ (null) | **$+2.45$ pts / $+0.1448$ (SUPPORTED)** | **0.069 / 0.235** |
| **Test — Held-out** (1409) | **97.3% / 0.1001** | **96.7% / 0.1247** | 94.7% / 0.2079 | 91.7% / 0.4710 | **96.5% / 0.1248** | $+0.21$ pts / $+0.0001$ (null) | **$+1.99$ pts / $+0.0832$ (SUPPORTED)** | **0.060 / 0.230** |
| **Test OOD — Held-out** (2000) | **93.7% / 0.2097** | **85.9% / 0.9331** | 85.1% / 0.9149 | 82.8% / 1.1570 | **85.3% / 0.7806** | $+0.60$ pts / **$-0.1525$ (OPPOSITE in CE)** | $+0.80$ pts / $-0.0182$ (null in CE) | **0.012 / 0.047** |

---

## 9. What we have learned (and what we have NOT shown)

**Learned**
1. **Always run controls.** A trained extra token improves loss with no visual information (soft-prompt effect). Without the shuffled, random and constant controls, step 8's "benefit" would have looked like a result.
2. **A benchmark must make the image necessary.** On ScienceQA the answer is mostly recoverable from text, so no image-based method can win. The first spatial dataset had the opposite problem: its images contained the answer.
3. **Alignment first.** A reader trained only on answers from a few thousand questions learns nothing; a few minutes of caption alignment on COCO makes the LLM read visual tokens (+1.03 nats true-vs-wrong).
4. **Leak-free sketches work for reading.** On v2 the reader uses image content (+8.1 points), and learns counting in `count_see` (21% → 64%).
5. **Imagination cannot add facts that are not in the text.** In the `see` track the text says nothing about the image, so a text-only student can never recover it beyond the prior. Imagination can only help by *re-representing what the text already states* (e.g. turning numbers into a to-scale picture), where the picture makes the reasoning easier.
6. **Text-only models are strong in-distribution and brittle out of it.** `triangle` text-only: 93%–94% in-distribution, 52% (chance) with bigger numbers. A scale-free picture stays at 98%.
7. **A single "scene" picture does not help state tracking** (`grid_walk`: 86.2 vs 85.6 on `val`, 90.1 vs 90.1 on `test`). The helpful visual there would be the board after each move (the PDF's per-step imagination), not the starting board.
8. **An MLP Imagination Head CAN learn example-specific visual representations from text hidden states (Phase B/C).** Across held-out test questions, `own` beats `own_shuffled` by $+1.99\%$ accuracy ($p < 0.05$, CI $[+1.21, +2.84]$) and retrieves the exact ground-truth diagram $60\times$ above chance (Top-1 6.0%, Top-5 23.0% among 1,409 diagrams). In-distribution, it accurately maps numeric dimensions in the text (`d=10`, `12x8`) to the corresponding visual features.
9. **The Single-Shot Imagination Bottleneck: Why `own` does not beat `const` or solve OOD `triangle` (Phase C).**
   - In-distribution, the LLM already solves the text at 96.5% (`const`), leaving only $+0.85\%$ headroom to the oracle (97.3%). `own` reaches 96.7% (and 95.6% vs 94.1% `const` on `triangle`), which is statistically indistinguishable from `const`.
   - Out-of-distribution (`test_ood`), where `triangle` has massive headroom (`const` 52.0% vs `oracle` 98.2%), `own` collapses to **51.7%**. Why? Because in the `oracle` condition, an **external Python renderer** did the scale-normalization (dividing the side lengths by the max length to draw the arcs), handing the visual reader a scale-invariant feature map where the arcs visibly intersect or miss. In the `own` condition, the Imagination Head must predict whether the arcs intersect *directly from the frozen LLM's question hidden state $h$*. Since $h$ cannot extrapolate the triangle inequality to large OOD numbers, the MLP head cannot either—and as the OOD gallery shows, it simply maps unfamiliar large numbers back to familiar in-distribution diagram embeddings.

**NOT shown yet**
- That self-generated visual tokens can significantly outperform a strong text-only baseline (`const`) on a reasoning task. (Requires a task where *iterative/compositional* construction of the visual state is easier than single-shot text reasoning, such as multi-step `grid_walk`.)
- Anything about the gate on a good label (the gate was only trained on a proxy).
- Anything about multi-step imagination or collapse across steps in the advanced architecture (needs the `grid_walk` chain).
- Anything about GRPO (steps 9 and 10 never run), real image decoding, or Visual CoT.
- Robustness across random seeds (all results are single runs; the CIs cover only question sampling).

**Caveats to remember**
- The triangle sketch nearly depicts the answer (the audit showed total arc length predicts it 99%), so "oracle 98%" is closer to "a solved diagram helps" than to "visual reasoning". And because the diagram is "pre-solved" by the renderer, single-shot imagination of that diagram is equivalent to solving the problem in one forward pass.
- 3×3 pooled CLIP features cannot read a ring or left/right relations (`mark_see` and `rel_see` stayed at chance). Whether 7×7 features can is untested because the grid-7 run lacked alignment.
- Wrong or OOD-mismatched visual tokens can hurt CE (on `test_ood`, `own` has higher/worse CE $0.9331$ than `const` $0.7806$ because distillation from a confident oracle makes the student overconfident even when its OOD imagination is wrong).

---

## 10. File versions, `old_progs`, and the run log

**The user's naming convention.** When a script is replaced, the previous copy goes to `old_progs/` with a suffix: `name_old.py`, `name_old2.py`, ... A **higher number means a more recent** copy; the plain filename is always the latest. A new copy may be saved even if the previous one was never run. Because commands were run with the plain filename at that moment, an older result was produced by `…_old`, `…_old2`, … in the order the file was *run*, and the most recent run corresponds to the plain filename.

**How to use this to find which file made which result:** count the runs of a script in the table below (oldest first). The 1st run's code is the oldest `old` copy that still matches, and so on. The exact copy depends on how many intermediate versions you saved, which cannot be determined from the chat alone.

**Incident to know about.** On 2026-10-24 at 11:17, `imagine_adv_lib.py`, `imagine_data.py` and `imagine_step8_advanced_a.py` were found to be **older copies than expected** (same timestamp, flagged by `imagine_check_env.py`). At 11:28, they were replaced with the newest versions (`imagine_data.py` 8213 bytes, `imagine_adv_lib.py` 14766 bytes, `LIB_VERSION = 4`), and `imagine_check_env.py` printed `"All consistent."` before Phase B and Phase C were executed.

### Run log (chronological, from the terminal timestamps)

| # | Approx. time | Script and arguments | Outcome in one line |
|---|--------------|----------------------|---------------------|
| 1 | 10-21 00:30 | `imagine_step1.py` | OK, d_model 576 |
| 2 | 00:33 | `imagine_step2.py` | OK, shapes correct |
| 3 | 00:34 | `imagine_step3.py` | OK, cosine 1.0 → −0.29 |
| 4 | ~00:36 | `imagine_step4.py` | OK, 3 checks pass |
| 5 | 00:40 | `imagine_step5.py` (first crash on CLIP `.bin`, fixed, rerun) | top-1 0.34 / top-5 0.69 |
| 6 | 00:51 | `imagine_step6.py` | gate acc 0.956, probes off |
| 7 | ~00:56 | `imagine_step7.py` | montage saved |
| 8 | 01:02 | `imagine_step8.py` | benefit 0.70, align 0.18 |
| 9 | 06:47 | `imagine_step8b.py` | information gain +0.0127 |
| 10 | 07:24 | `imagine_step8c.py` | own_info_gain +0.0126 |
| 11 | 08:51 | `imagine_step8d.py --n_eval 150` | not distinguishable |
| 12 | 15:57 | `imagine_step8_advanced_a.py --vis global` (ScienceQA) | 71.8 / 72.3 / 73.2 |
| 13 | 16:28 | `…_a.py --vis grid --grid 3 --epochs 3` | 70.3 / 70.3 / 70.0 |
| 14 | 16:45 | `…_a.py --vis grid --grid 3 --epochs 3 --w_contrast 1.0` | 71.5 / 71.8 / 71.7 |
| 15 | ~17:20 | `…_a0.py --vis grid --grid 3` | A0 passed, gap +1.03 |
| 16 | 18:00 | `…_a.py --vis grid --grid 3 --epochs 3` (aligned reader) | 69.5 / 69.2 / 68.5 |
| 17 | 10-21 23:xx | `imagine_diagnose_data.py`; `imagine_build_spatial_cot.py --n_train 80000 --n_val 4000 --n_test 4000` | ScienceQA text-solvable; v1 built |
| 18 | 10-21 23:46 | `…_a.py --data spatial --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4 --w_contrast 1.0` | 100 / 58.3 / 99.2 (leaky) |
| 19 | 10-22 09:3x | `imagine_audit_data.py` on v1 (8/8 flagged); `imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000`; audit on v2 (clean) | v2 built |
| 20 | 10-22 09:35 | `…_a.py --data spatial2 --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4` | **81.8 / 73.8 / 76.9** |
| 21 | 10-23 23:59 | `…_a_eval.py --data spatial2 --vis grid --grid 3 --split test_ood --n 2000` | OOD 93.7 / 82.8 / 85.3 |
| 22 | 10-24 00:02 | `…_a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4` | 76.2 / 75.6 / 76.1 (no A0) |
| 23 | 10-24 (crashes) | `…_b.py`, `…_c.py` | `NameError: suffix_of`, then `TypeError: root`: version mismatch (Section 11) |
| 24 | 10-24 11:17 / 11:28 | `imagine_check_env.py` | 5 items OLD at 11:17; replaced at 11:28 $\to$ `All consistent.` |
| 25 | **10-24 11:30** | `…_b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3` | **Completed:** val `own` 95.2% vs `shuffled` 92.8% (+2.45 pts SUPPORTED), `const` 95.1%, top-1 retrieval 6.9% |
| 26 | **10-24 12:36** | `…_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test --n_test 2000` | **Completed (`test`):** `oracle` 97.3%, `own` 96.7%, `const` 96.5%, `shuffled` 94.7% (+1.99 pts SUPPORTED), top-1 6.0% |
| 27 | **10-24 12:40** | `…_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000` | **Completed (`test_ood`):** `oracle` 93.7%, `own` 85.9%, `const` 85.3%, `shuffled` 85.1%; OOD `triangle` `own` 51.7% (chance) |

---

## 11. Environment problems met and how they were fixed

| Problem | Cause | Fix |
|---------|-------|-----|
| `ValueError: ... upgrade torch to at least v2.6` when loading CLIP | The `.bin` weights are refused on torch < 2.6 | Load CLIP with `use_safetensors=True` (in `imagine_data.py`) |
| `Dataset scripts are no longer supported` (nlphuji COCO) | Newer `datasets` dropped script-based datasets | The loader tries several sources; `lmms-lab/COCO-Caption` worked |
| `NameError: suffix_of` then `TypeError: get_spatial_cot() got an unexpected keyword argument 'root'` | An older `imagine_adv_lib.py` / `imagine_data.py` in the folder than the scripts expected | Lib version 4 is self-contained for the spatial loader; `imagine_check_env.py` checks consistency; replace outdated files |
| Grid-7 run did not start from the aligned reader | A0 checkpoints are stored per grid size, and no grid-7 A0 existed | Run A0 for grid 7 first (`imagine_step8_advanced_a0.py --vis grid --grid 7`) |
| Windows symlink warning, TensorFlow oneDNN message, `torch.load weights_only` FutureWarning, `hf_xet` notice | Harmless | Ignore |

---

## 12. Current status and next steps

**Current status (as of 2026-10-24 12:45):**
Both **Phase B** and **Phase C (`test` and `test_ood`)** have finished cleanly! This completes the entire single-step answer-first advanced pipeline (`A0 -> A -> B -> C`).

**Summary of the Phase B / C Verdict:**
We tested three possible outcomes for Phase C:
1. *`own > blank` on OOD `triangle`*: Did **not** happen (`own` = 51.7%, `const` = 52.0%, `oracle` = 98.2%).
2. *`own ≈ blank` in accuracy*: **Happened.** On `test`, `own` is 96.7% vs `const` 96.5% ($+0.21$ pts, not distinguishable from 0; on in-distribution `triangle`, `own` is 95.6% vs `const` 94.1%). On `test_ood`, `own` is 85.9% vs `const` 85.3% ($+0.60$ pts accuracy, driven by `fit_ball_box` 99.2% vs 97.6% and `grid_walk` 76.8% vs 75.8%).
3. *`own < blank` in CE on OOD*: **Happened on `test_ood` CE** ($0.9331$ vs $0.7806$, `OPPOSITE`), because KL-distillation (`kd`) trained the student to mimic the teacher's near-100% confidence; when OOD `triangle` imagination fails at 51.7% accuracy, those confident wrong predictions heavily penalize cross-entropy.
4. *`own > own_shuffled` and image retrieval*: **Strongly SUPPORTED** in-distribution ($+2.45$ pts on `val`, $+1.99$ pts on `test`, Top-5 retrieval 23.0% vs 0.4% chance). The Imagination Head truly learns to map text descriptions to their specific visual tokens, but a single one-shot MLP projection from the question hidden state $h$ cannot compute scale-free OOD geometry that $h$ itself does not know.

**Planned next experiments (in suggested order)**
1. **Multi-step `grid_walk` chain** (the PDF's original per-step loop, and now the most logical next step):
   - *Why:* Single-shot imagination failed to beat `const` because generating a finished diagram in one shot requires already knowing the final state/geometry. In `grid_walk`, each individual move (`up`, `left`, `down`) is a trivial local spatial update, whereas tracking 5–8 moves in text is hard (`const` is 85.6%–90.1% in-distribution and drops to **75.8%** on `test_ood`).
   - *How:* Generate step images (`imagine_build_spatial_v2.py --step_images`), train a step-wise transition / imagination loop that updates the board tokens after each move from the previous imagined board + current move, and answer from the final imagined board. Also measure anti-collapse across steps.
2. **End-to-end LoRA + Imagination Head fine-tuning (or temperature scaling on `kd`):** In Phase B, the LLM and reader were strictly frozen while training the head. Allowing LoRA or the reader to co-adapt to noisy/imperfect imagined tokens (or lowering the KL distillation weight on uncertain OOD inputs) would prevent the CE overconfidence penalty seen on `test_ood`.
3. **Redo grid 7 properly (A0 for grid 7, then Phase A):** Run `imagine_step8_advanced_a0.py --vis grid --grid 7` first, then Phase A, to see whether $7\times 7$ patch tokens can read the black ring (`mark_see`) and spatial relations (`rel_see`).
4. **Photo VQA (A-OKVQA)** as a second, natural-image domain.
5. **Gate training with a better label** (using an LLM judge via an API such as Gemini).
6. **GRPO (steps 9 and 10)**, adapted to the advanced track, with answer accuracy as reward (allowing the imagination head to discover helpful latent tokens via RL rather than purely imitating CLIP teacher features).
7. **The user's dual-stack idea** (text stack plus latent-image stack with bidirectional cross-attention) as an architecture upgrade, enabling iterative layer-by-layer visual construction rather than predicting the image in a single MLP pass at the final layer.
8. **More seeds and larger models** (Qwen2.5-0.5B is about the practical limit on 6 GB VRAM).

---

## 13. Command cheat-sheet

```powershell
# sanity: are my files consistent?
python imagine_check_env.py

# datasets
python imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000
python imagine_audit_data.py imagine_cache/spatial_cot_v2/val.json

# alignment (once per grid size) and Phase A
python imagine_step8_advanced_a0.py --vis grid --grid 3
python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4

# evaluation without training (where is the headroom?)
python imagine_step8_advanced_a_eval.py --data spatial2 --vis grid --grid 3 --split test_ood --n 2000

# Phase B (imagination head) and Phase C (final test - BOTH COMPLETED)
python imagine_step8_advanced_b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test --n_test 2000
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000

# Optional next: redo grid 7 with proper A0 alignment first
python imagine_step8_advanced_a0.py --vis grid --grid 7
python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4
```

*Out-of-memory tip: add `--micro_bs 4 --accum 4`. Interrupted Phase A run: add `--resume`.*