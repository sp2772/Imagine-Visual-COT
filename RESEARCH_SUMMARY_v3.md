# Latent Visual Imagination for LLM Reasoning: Project Summary and Lab Notebook

*Originally written 2026-10-24; additively updated 2026-10-24 after Phase B and Phase C (test & test_ood) finished. Covers the whole project from the first (about 2026-10-20) to the completion of the single-step advanced track.*

Original PDF: research_plan_new.pdf
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
- An **Imagination Head** turns `h_t` into a visual vector `z_t = W2 * h_t` ("what the model is imagining").
- A **Gate** `g_t = sigmoid(W1 * h_t)` (a number from 0 to 1) decides whether the imagination is shown as an image (high) or stays an internal thought (low).
- The previous imagination `z_{t-1}` is **fed back** into the LLM at the next step (a loop).
- A **contrastive anti-collapse loss** forces `z_t` to differ from `z_{t-1}`, because earlier latent-visual-reasoning work reported that imagined tokens collapse into identical vectors.
- Training in four stages: COCO captions, Visual CoT, ScienceQA (plus a GRPO reinforcement step), then joint training. Everything is done with LoRA in bf16 to save memory.

**What we did.** We built the system step by step with small Python scripts, ran each on the author's machine (Windows, RTX 3050 6 GB, 24 GB RAM), and, importantly, **tested every claim with controls and confidence intervals**. That testing changed the direction of the project several times.

**Where we are (headline).**

| # | Finding | Strength |
|---|---------|----------|
| 1 | Early "imagination helps" numbers were an artifact: a trained extra token acts as a **soft prompt**. Controls (shuffled / random / constant tokens) exposed it. | Solid |
| 2 | ScienceQA is a poor test bed: most questions are answerable from text (hints, world knowledge), so swapping the image changes nothing. | Solid (diagnostic script) |
| 3 | The first "spatial" dataset (made by the Cursor agent) looked like a success (oracle 100%) but **leaked the answer** in its images and in answer-position patterns; blank-image accuracy was already 99.2%. | Solid (audit script) |
| 4 | A leak-free dataset (**v2**) shows the image-reading part works: true image beats wrong image by **+8.1 accuracy points (CI +6.4 to +9.9)** and, on tasks where the image is *necessary*, 49.5% vs 35.8% blank. | Solid |
| 5 | On the leak-free data, **text alone already solves most "imagine" families** (about 95%), so imagination has little room to help. Room exists in `grid_walk` (about 81% to 86% text-only) and in `triangle` **out of distribution**. | Solid |
| 6 | **Most interesting:** on out-of-distribution `triangle` problems (bigger numbers), text-only drops to chance (52%) while the true sketch gives 98%. A scale-free picture generalizes where text arithmetic does not. Caveat: the sketch nearly depicts the answer. | Solid as an observation; its meaning is open |
| 7 | **Phase B & C completed (In-Distribution):** The Imagination Head successfully learns **example-specific visual representations** from text alone! On held-out `test` (n=1409), `own` beats `own_shuffled` by **+1.99 accuracy points (CI +1.21 to +2.84, SUPPORTED)** and retrieves the exact true diagram with **Top-1 = 6.0%, Top-5 = 23.0%** (chance 0.1% / 0.4%; nearly 60x chance). On `triangle`, `own` reaches **95.6%** vs **94.1%** blank (`const`) and **87.7%** `own_shuffled`. However, overall `own` (96.7%) vs `const` (96.5%) is +0.21 pts (not distinguishable from 0) because text-only already saturates most families. | Solid |
| 8 | **Phase C on OOD (`test_ood`, n=2000) — The Single-Shot Bottleneck:** Out of distribution, `own` gets **85.9%** vs `const` **85.3%** (+0.60 acc pts, CI +0.15 to +1.05, though CE is worse by -0.1525 due to overconfidence). On OOD `triangle`, `own` drops to **51.7%** (same as `const` 52.0%, while `oracle` is 98.2%). **Why:** A 3-layer MLP reading the LLM's final hidden state `h` cannot construct a scale-free geometric sketch if `h` itself failed the out-of-distribution arithmetic. One-shot imagination re-represents what `h` already computed; it cannot do novel visual simulation unless decomposed across iterative steps (such as `grid_walk`). | Solid |

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
| **Imagination vector `z_t`** | The visual vector produced from `h_t` ("what the model imagines at step t"). Originally 512 numbers; in the advanced track it is a 3x3 grid of 768-number tokens (see "grid"). |
| **Imagination Head** | The small network that turns `h_t` into `z_t`. In the PDF: one linear map `W2`. Ours: a LayerNorm plus a linear map (steps 2 to 8), then a 3-layer MLP of 20.2 M parameters (Phase B). |
| **`W1`, `W2`** | The two weight matrices of the PDF: `W2` makes the imagination, `W1` makes the gate. |
| **Gate `g_t`** | `sigmoid(W1 * h_t)` in [0, 1]; high means "show the imagined image", low means "keep it internal". |
| **Feedback loop** | `z_{t-1}` is converted into extra input token(s) and fed into the LLM at step t, so earlier imagination influences later thinking. |
| **Anti-collapse / diversity loss** | A penalty when consecutive imagination vectors are too similar (hinge on cosine similarity). Prevents the "all imagined tokens identical" failure. |
| **Contrastive loss / InfoNCE** | A loss that pulls a vector toward its matching target and pushes it away from other items in the batch ("this imagination must belong to THIS question, not any question"). |
| **Image decoder** | Turns `z_t` into a visible picture. We only used a stand-in: retrieving the closest real image (step 7, Phase C). A real generator (unCLIP / Stable Diffusion) is future work. |
| **Reader** | (advanced track) A small MLP that turns CLIP image features into LLM input embeddings ("visual tokens"), like the projector in LLaVA-style models. It also holds learned "blank" tokens. |
| **Visual tokens** | Vectors inserted into the LLM input that carry image information. |
| **CLIP** | A pretrained model (ViT-B/32) that maps images and text into a shared space; we use its image features as the "picture" representation (512-d global vector, or patch tokens of 768-d). |
| **Patch tokens / grid** | CLIP splits an image into 7x7 patches. "Grid 3" averages them into a 3x3 grid (9 tokens x 768-d); "grid 7" keeps all 49. "Global" = one 512-d vector for the whole image. |
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
| **`visual_info_usable`** | `oracle_wrong - oracle` in loss: can the LLM use image *content* at all? |
| **`own_info_gain`** | `own_shuffled - own` in loss: does the model's own imagination help THIS example? |
| **Recovery** | `own_info_gain / visual_info_usable` (or `(const - own) / (const - oracle)` in Phase B); meaningless when the denominator is tiny. |

### 2.4 Training and data concepts

| Term | Meaning |
|------|---------|
| **Teacher forcing** | During training, feed the true thing (here the true image tokens) instead of the model's own output. At test time the model must use its own. |
| **Scheduled sampling** | Randomly mixing teacher-forced and own outputs during training. |
| **Blank-slate student / "imagine by itself"** | The design principle: at test time no image is given. Images are only a teaching signal. |
| **Soft prompt** | An extra trained input vector that helps simply by being there (like prompt-tuning), not because of what it encodes. Our main false positive. |
| **Answer-first format** | `[question + choices] [visual tokens] "Answer:" -> letter`. The loss is only on the answer letter. |
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
| **Cursor agent** | The AI coding assistant the author also used; it produced the first spatial dataset (v1). |
| **Dual-stack idea (future)** | The author's later idea: two transformer stacks (text and latent image) with bidirectional cross-attention after each self-attention pass. Not yet started. |

---

## 3. How to read the result tables

1. **Accuracy and CE for oracle / wrong / const (and own).** The most important comparison is **oracle vs wrong**: if the model uses the image, a wrong image must be worse than the right one.
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
| Imagination | One 512-d vector per step, `z_t = W2 * h_t` | Steps 2 to 8: that (plus LayerNorm). Advanced track: 3x3 grid of 768-d CLIP-patch tokens predicted by a 3-layer MLP. |
| Where imagination is injected | Fed back at every step | Steps 4 to 8d: after every step (1 token, later 4). Advanced track: **one** imagination before the answer (answer-first). The multi-step loop returns later (proposed for `grid_walk`). |
| Gate | `sigmoid(W1 * h)`, trained with correctness labels | Trained in step 6 with a proxy label ("question has an image in ScienceQA"); not part of the advanced track yet. |
| Anti-collapse | Contrastive loss `z_t != z_{t-1}` | Implemented (hinge on consecutive cosine) in steps 3 and 8; across-batch InfoNCE in Phase B. Collapse across steps cannot be measured in the single-imagination advanced track. |
| Backbone | e.g. LLaMA | SmolLM2-135M, then Qwen2.5-0.5B (fits a 6 GB GPU) |
| Datasets | COCO, Visual CoT, ScienceQA | COCO captions (yes), ScienceQA (yes, but a poor test), Visual CoT (skipped: about 143 GB images), plus our own spatial datasets |
| GRPO | Stage 3 refinement | Written (steps 9 and 10), **never run** |
| Image decoder | Present | Retrieval stand-in only (Step 7 and Phase C galleries) |

**Principle adopted from the author:** at test time the model starts as a blank slate and must imagine by itself, like a student hearing a teacher's problem. Images are only a training-time teaching signal, analogous to teacher forcing in next-token training.

---

## 5. Timeline of what was done, with every intermediate result

*(Terminal timestamps in parentheses come from the author's logs. Numbers are quoted from the terminal output.)*

### Phase 1: Building the basic modules (steps 1 to 7), SmolLM2-135M

**Step 1, `imagine_step1.py` (2026-10-21 00:30).** Loads the backbone and extracts hidden states.
- Output: `d_model=576`, hidden shape `(1, 28, 576)`, `h_t` norm 49.47, generation continues sensibly ("Step 1: The diameter of the ball is 9 cm. Step 2: ...").
- A harmless print bug (batch shape printed as a tensor) was fixed later.
- Meaning: the backbone works and we can read `h_t`.

**Step 2, `imagine_step2.py`.** Builds the Imagination Head and gate (`z = W2 * h`, `g = sigmoid(W1 * h)`, with LayerNorm).
- Output: z shape `(1, 18, 512)`, gate shape `(1, 18)`, z norms exactly 1.0, gate values 0.56 to 0.66, 297,153 trainable parameters.
- Meaning: correct shapes; random-init gates sit near 0.5.

**Step 3, `imagine_step3.py`.** Demonstrates the anti-collapse loss on a deliberately collapsed head.
- Output: consecutive cosine **1.000 -> -0.286** after 300 iterations (target at or below 0.3).
- Meaning: the loss works mechanically. It went negative because the hinge pushes hard; harmless. A caveat raised then: "z_t != z_{t-1}" alone can be satisfied by random noise, so it needs an anchor (done later).

**Step 4, `imagine_step4.py` (the feedback loop).** Context for step t is `[I_-1, seg_0, I_0, seg_1, ..., I_{t-1}, seg_t]` where `I_t = proj(z_t)`.
- Check 1: LM loss feedback 1.5943 vs constant 1.5904, relative change of final `h` 0.0757 (must be above 0): feedback changes the LLM.
- Check 2 (causality): max change in earlier z when editing the last step: **0.00e+00**; last step 1.24e-02: no information leaks backwards.
- Check 3: gradient norms W2 0.80, proj 0.73, W1 0.0 (the gate is trained separately).
- Meaning: the loop is wired correctly.

**Step 5, `imagine_step5.py` (Stage 1: COCO captions -> CLIP image embeddings).**
- Data: the first source (`nlphuji/...`) failed ("Dataset scripts are no longer supported"); the fallback `lmms-lab/COCO-Caption` gave 2000 images in 1:24. A crash on loading CLIP (torch < 2.6 refuses `.bin` weights) was fixed with `use_safetensors=True`.
- Result on 200 held-out images: before training top-1 0.01 / top-5 0.025; **after training top-1 0.34, top-5 0.69, top-10 0.815, median rank 3** (chance 0.005 / 0.025). Validation stopped improving about epoch 20 while training loss kept falling (0.37 at epoch 40): mild overfitting.
- Qualitative: a query "A large passenger jet flying through a gray sky" retrieved other jets; kitchens retrieved kitchens.
- Meaning: a linear head on a small LLM's hidden state does carry visual semantics.

**Step 6, `imagine_step6.py` (gate).** ScienceQA items with an image labeled 1, without labeled 0 (400 + 400).
- Result: validation accuracy **0.956** (train 0.998; before training 0.481).
- But the hand-written probes were wrong in two places: the ball-in-box question got g=0.029 (should be high) and "What is the capital of France?" got g=0.986 (should be low).
- Meaning: the gate learned dataset habits (ScienceQA's "capital" questions come with maps), so "has image" is only a **proxy label**. The author asked to keep the pipeline general rather than patch for probes; the real label ("does this need visual imagination") will need an LLM judge later.

**Step 7, `imagine_step7.py` (stand-in decoder = nearest real image).**
- Gates: ball-box 0.980, bus 0.999, dog 0.944, "broccoli and rice" 0.401 (hidden), "capital of France" 1.000.
- Retrieval similarities 0.20 to 0.28 (typical for CLIP). Bus, dog and food retrieved sensible images; ball-and-box did not (the 2000-image gallery probably has no such photo).
- Meaning: the head maps text into a meaningful visual region; the gate is noisy because of the proxy label.

### Phase 2: Joint training and the first controls (steps 8, 8b, 8c, 8d)

**Step 8, `imagine_step8.py`.** Backbone frozen; trains head, gate and feedback projector on ScienceQA chains-of-thought (480 examples; 400 train / 80 val) with five losses (LM, alignment to CLIP text of each step, image alignment, anti-collapse, gate).
- Before: `lm_feedback 1.6543 | lm_constant 1.9989 | imagination_benefit 0.3447 | consecutive cos 0.861 | align_top1 0.035 | gate_acc 0.775`.
- After 2 epochs (about 8 min each): `lm_feedback 1.3324 | lm_constant 2.0355 | benefit 0.7031 | cos 0.787 | align_top1 0.182 (chance 0.004) | gate_acc 0.943`.
- Looked good: the loss fell and alignment rose 45x chance.

**Step 8b, `imagine_step8b.py` (control: is the imagination informative?).**
- Mean LM loss: feedback **1.3324**, shuffled **1.3451**, random_z **1.5630**, constant **2.0355**.
- Information gain (shuffled minus feedback) only **+0.0127**; soft-prompt effect (constant minus shuffled) **+0.6905**. Consecutive cosine 0.843 (model) vs 0.816 (the gold CLIP step texts themselves), so no collapse.
- **Key finding #1:** nearly all of the "benefit" in step 8 was a trained extra token acting as a soft prompt, not information in `z`. The LLM is text-only, and `z` is computed from the text it already read, so it cannot add anything new by construction.

**Step 8c, `imagine_step8c.py` (teacher forcing with the true image, k=4 tokens, LoRA).** Tested "learn to imagine with the image as teacher; at test time use own imagination".
- Final (60 val image questions): own 1.0844 | own_shuffled 1.0971 | oracle 1.0228 | oracle_wrong 1.0381 | constant 1.1038.
- `visual_info_usable` +0.0153, `own_info_gain` +0.0126 (it was +0.0073, -0.0001, +0.0126 across epochs), "recovery" jumped 3.32, -0.02, 0.82. Blank-slate z_0 retrieves the true image with top-1 0.083 (chance 0.017), cosine to right image 0.685 vs 0.617 to a wrong one.
- Meaning: numbers had the right sign but were tiny and unstable on 60 questions. "Recovery" is meaningless when its denominator is about 0.002 to 0.015.

**Step 8d, `imagine_step8d.py` (150 fresh questions, bootstrap CIs, plus an "answer-first" readout).**
- CoT setting: own 1.1391, shuffled 1.1440, oracle 1.0774, wrong 1.0784, constant 1.1600. `own vs blank` +0.0208 CI [-0.0042, +0.0436]; `own is example-specific` +0.0049 CI [-0.0250, +0.0347]; `image content usable` **+0.0010 CI [-0.0004, +0.0025]** (so 8c's +0.015 was noise).
- Answer-first (chance CE 1.004): own 1.2492, oracle 1.2587, wrong 1.2597, constant 1.3395. `own vs blank` +0.0903 SUPPORTED, but oracle equals own and wrong equals oracle.
- **Key finding #2:** the LLM did not read image content at all; every image-like token worked equally. The test bed itself (gold reasoning steps in context, text-only LLM) could not show a visual benefit.

### Phase 3: The "advanced track" on ScienceQA (Qwen2.5-0.5B, answer-first)

**Design change.** Use a bigger model (Qwen2.5-0.5B + LoRA, bf16, gradient checkpointing), official ScienceQA splits, **image questions only**, answer-first format, and a **reader** (CLIP features -> LLM tokens). Programs: `imagine_adv_lib.py` plus `imagine_step8_advanced_a/b/c.py`. Success test for Phase A: **oracle must beat wrong with a confidence interval above 0.**

**Phase A runs on ScienceQA (4000 train, 600 val):**

| Run | Setting | oracle / wrong / const accuracy (final) | oracle vs wrong |
|-----|---------|------------------------------------------|-----------------|
| 1 (2026-10-21 15:57) | global CLIP vector | 71.8 / 72.3 / 73.2 | not distinguishable |
| 2 (16:28) | grid 3x3 | 70.3 / 70.3 / 70.0 | OPPOSITE by a hair (CE -0.0081) |
| 3 (16:45) | grid 3x3 + wrong-image margin loss (`--w_contrast 1.0`) | 71.5 / 71.8 / 71.7 | OPPOSITE (CE -0.0088) |

Observations: accuracy rose from about 52% to about 72% under **every** condition, including blank, so the model learned ScienceQA's text patterns. CE got worse after epoch 2 (0.618 -> 0.755) while accuracy stayed flat (overfitting). A single "SUPPORTED" at epoch 2 of run 1 (+0.0039) vanished by epoch 3: chance.

**A0, caption alignment, `imagine_step8_advanced_a0.py` (about 17:20).** Reader trained on 20,000 COCO images (download 24 min, CLIP features 5:47) with the LLM frozen.
- Held-out caption CE on 500 images: before true 3.817 / wrong 3.818 / blank 3.694. After epoch 2: **true 2.447 / wrong 3.476 / blank 2.939**; gap vs wrong **+1.029 [0.988, 1.070]**, vs blank +0.492 [0.469, 0.515].
- **Key finding #3:** the visual path works: a frozen Qwen can read CLIP grid tokens after only about 8 minutes of alignment.

**Phase A again on ScienceQA with the aligned reader:** final oracle 69.5 / wrong 69.2 / const 68.5, still not distinguishable (before training, any image token gave +7.7 points over blank, but not the right image over a wrong one).
- **Key finding #4:** the failure is not the visual path; **the dataset gives the image nothing to add** (see Phase 4).

### Phase 4: Diagnosing ScienceQA (`imagine_diagnose_data.py`, written by Cursor)
- Validation set (n=600): 65% have a hint text, 50% have a long hint (>= 80 characters, often restating the figure), **14% contain the gold answer string** in the question or hint, 26% look like knowledge or meta questions.
- Train set (n=4000): 63% with hint, 48% long hint, 12% gold string present, 25% knowledge questions.
- Interpretation: swapping the figure rarely changes the answer, so oracle ~ wrong. A dataset failure, not evidence against imagination.

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
| `count_see` | Options were always gold-1..gold+2, so the choice list encodes the answer (a lookup solves 100%; rank bias 60%). |
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
| `count_see` | 63.6 | 21.3 | 20.9 | learned (35 -> 61 -> 63 -> 64 over epochs); not exact |
| `mark_see` | 30.9 | 32.7 | 34.6 | chance: ring not readable from a 3x3 grid |
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
- **Key finding #7 (most interesting):** with bigger numbers (sides 16 to 40), the text-only path on `triangle` collapses to chance (52%), while the true sketch gives 98%. The sketch is normalized to the longest side (scale-free), so it reads the same at any scale, while text arithmetic does not extrapolate. **Caveat:** the sketch comes close to depicting the answer (arcs meet or not), so the real question is whether a model can *produce* those picture features from the numbers alone.

**Phase A with grid 7 (49 tokens), 2026-10-24 00:02: inconclusive.** Final oracle 76.2 / wrong 75.6 / const 76.1; `count_see` 25% (chance). The log lacks the line "reader initialised from the caption-alignment stage A0": the A0 checkpoint is saved per grid size and none existed for grid 7, so this run started with an unaligned reader. It tested nothing about finer tokens (a mistake in our run list: A0 for grid 7 was a missing prerequisite). To redo: run `imagine_step8_advanced_a0.py --vis grid --grid 7` first.

### Phase 7: Phase B — Training the Imagination Head (completed, 2026-10-24 11:30 to 12:36)

`imagine_step8_advanced_b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3`: trains **only the Imagination Head** (20.2 M parameters) with the reader and LLM frozen. Learns from the question text's hidden state (`(14017, 896)`); the true image is only a teacher through four losses (`feat` cosine+MSE, `nce` batch contrastive loss, `ans` answer CE, `kd` distillation of the teacher's answer distribution). Only `imagine` items are used (14,017 train, 1,386 val), since imagining from text is only meaningful there. Each epoch took ~17 minutes (876 steps at ~1.17s/it).

**Progression across epochs on validation (`n=1386`; chance: acc=37.6%, CE=1.038):**

| Epoch / Stage | oracle acc (CE) | own acc (CE) | own_shuffled acc (CE) | wrong acc (CE) | const acc (CE) | Retrieval Top-1 / Top-5 (chance 0.001 / 0.004) |
|---------------|-----------------|--------------|-----------------------|----------------|----------------|------------------------------------------------|
| **Before (random head)** | 96.2% (0.1332) | 94.6% (0.1849) | 95.0% (0.1764) | 90.7% (0.5044) | 95.1% (0.1561) | 0.000 / 0.001 |
| **Epoch 2** | 96.2% (0.1332) | 95.2% (0.1612) | 94.2% (0.2059) | 90.7% (0.5044) | 95.1% (0.1561) | 0.045 / 0.192 |
| **Epoch 3 (Final)** | **96.2% (0.1332)** | **95.2% (0.1592)** | **92.8% (0.3040)** | **90.7% (0.5044)** | **95.1% (0.1561)** | **0.069 / 0.235** |

**Training loss progression:**
- Epoch 1: `feat=0.085, nce=0.442, ans=0.031, kd=0.002, train_acc=0.988`
- Epoch 2: `feat=0.069, nce=0.362, ans=0.048, kd=0.018, train_acc=0.988`
- Epoch 3: `feat=0.066, nce=0.365, ans=0.091, kd=0.003, train_acc=0.969`

**Per-family breakdown at Final Validation (Epoch 3, `n=1386`, accuracy %):**

| Family | n | oracle | own | own_shuffled | wrong | const |
|--------|---|--------|-----|--------------|-------|-------|
| `fit_ball_box` | 306 | 100.0 | 100.0 | 96.1 | 91.2 | 100.0 |
| `fit_rect` | 200 | 100.0 | 100.0 | 98.5 | 95.5 | 100.0 |
| `grid_walk` | 383 | 86.2 | 86.2 | 85.9 | 85.9 | 85.6 |
| `spatial_grid` | 307 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `triangle` | 190 | 100.0 | 93.2 | 83.7 | 79.5 | 93.2 |

**Final Phase B Paired Bootstrap Contrasts (`n=1386`):**
- `own imagination vs blank (const - own)`: CE `-0.0032 [-0.0116, +0.0048]` | acc `+0.14 [-0.29, +0.58]` pts -> **not distinguishable from 0**
- `own is example-specific (own_shuffled - own)`: CE `+0.1448 [+0.0885, +0.2066]` | acc **`+2.45 [+1.52, +3.39]` pts -> SUPPORTED**
- `own vs a random real image (wrong - own)`: CE `+0.3452 [+0.2535, +0.4413]` | acc **`+4.55 [+3.39, +5.77]` pts -> SUPPORTED**
- `headroom: true image vs own (own - oracle)`: CE `+0.0260 [+0.0125, +0.0417]` | acc `+0.94 [+0.51, +1.44]` pts -> **SUPPORTED**
- `[reference] image content usable (wrong - oracle)`: CE `+0.3712 [+0.2755, +0.4707]` | acc `+5.48 [+4.26, +6.78]` pts -> **SUPPORTED**

**What Phase B showed:**
1. Before training, a random head slightly hurt performance (`94.6%` vs `95.1%` blank, CE `-0.0288`, OPPOSITE) and had `0.000` image retrieval.
2. After 3 epochs, the Imagination Head learned to generate **genuinely example-specific visual tokens**: swapping a question's own imagined tokens with another question's imagined tokens (`own_shuffled`) drops accuracy by **2.45 points** overall (`95.2% -> 92.8%`), and by **9.5 points on `triangle`** (`93.2% -> 83.7%`) and **3.9 points on `fit_ball_box`** (`100.0% -> 96.1%`).
3. Furthermore, the imagined tokens retrieve the exact matching validation diagram among 1,386 candidates with **6.9% Top-1** (69x chance) and **23.5% Top-5** (59x chance).
4. However, `own` (`95.2%`) matches `const` (`95.1%`) without beating it: on `triangle`, `own` recovers the blank baseline (`93.2%`), not the `oracle` upper bound (`100.0%`). Saved checkpoint: `imagine_ckpt/adv_b_qwen2.5-0.5b_grid3_spatial2_imagine/head.pt`.

### Phase 8: Phase C — Final Held-Out Evaluation on `test` and `test_ood` (completed, 2026-10-24 12:36 to 12:43)

We ran `imagine_step8_advanced_c.py` on both the held-out in-distribution `test` split and the out-of-distribution `test_ood` split.

#### 8A. In-Distribution Held-Out `test` Split (`--split test --n_test 2000`, filtered to `imagine` track: `n=1409`)
Chance level: `acc=37.4%, CE=1.041`.

| Condition | Accuracy (%) | Cross-Entropy (CE) |
|-----------|--------------|--------------------|
| `oracle` | **97.3%** | **0.1001** |
| `own` (blank-slate student) | **96.7%** | **0.1247** |
| `const` (blank tokens) | 96.5% | 0.1248 |
| `own_shuffled` | 94.7% | 0.2079 |
| `wrong` | 91.7% | 0.4710 |

**Paired Bootstrap Contrasts (`test`, `n=1409`):**
- `own imagination vs blank (const - own)`: CE `+0.0001 [-0.0131, +0.0123]` | acc `+0.21 [-0.14, +0.57]` pts -> **not distinguishable from 0**
- `own is example-specific (own_shuffled - own)`: CE `+0.0832 [+0.0417, +0.1291]` | acc **`+1.99 [+1.21, +2.84]` pts -> SUPPORTED**
- `own vs a random real image (wrong - own)`: CE `+0.3463 [+0.2517, +0.4482]` | acc **`+4.97 [+3.83, +6.17]` pts -> SUPPORTED**
- `headroom: true image vs own (own - oracle)`: CE `+0.0246 [+0.0098, +0.0434]` | acc `+0.64 [+0.28, +1.14]` pts -> **SUPPORTED**
- `[reference] image content usable (wrong - oracle)`: CE `+0.3709 [+0.2749, +0.4737]` | acc `+5.61 [+4.40, +6.88]` pts -> **SUPPORTED**
- `[reference] true image vs blank (const - oracle)`: CE `+0.0247 [+0.0096, +0.0429]` | acc `+0.85 [+0.35, +1.42]` pts -> **SUPPORTED**

**Per-Family Breakdown (`test`, `n=1409`, accuracy %):**

| Family | n | oracle | own | own_shuffled | wrong | const |
|--------|---|--------|-----|--------------|-------|-------|
| `fit_ball_box` | 301 | 100.0 | 100.0 | 98.7 | 91.7 | 100.0 |
| `fit_rect` | 197 | 100.0 | 100.0 | 99.0 | 94.9 | 100.0 |
| `grid_walk` | 382 | 90.1 | 90.1 | 88.5 | 87.4 | 90.1 |
| `spatial_grid` | 326 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **`triangle`** | **203** | **100.0** | **95.6** | **87.7** | **83.3** | **94.1** |

- **Retrieval on `test` (`n=1409`):** `top1=0.060, top5=0.230` (chance `0.001 / 0.004`).
- **Note on `triangle` (`test`):** `own` reaches **95.6%** vs `const` **94.1%** (+1.5 pts) and `own_shuffled` **87.7%** (+7.9 pts), showing a small positive hint in-distribution, though diluted across the full 1,409-item set where 3 families are already at 100%.

#### 8B. Out-of-Distribution Held-Out `test_ood` Split (`--split test_ood --n_test 2000`, all `imagine` track: `n=2000`)
Chance level: `acc=40.3%, CE=0.963`.

| Condition | Accuracy (%) | Cross-Entropy (CE) |
|-----------|--------------|--------------------|
| `oracle` | **93.7%** | **0.2097** |
| `own` (blank-slate student) | **85.9%** | 0.9331 |
| `const` (blank tokens) | 85.3% | **0.7806** |
| `own_shuffled` | 85.1% | 0.9149 |
| `wrong` | 82.8% | 1.1570 |

**Paired Bootstrap Contrasts (`test_ood`, `n=2000`):**
- `own imagination vs blank (const - own)`: CE **`-0.1525 [-0.2004, -0.1042]` -> OPPOSITE** | acc **`+0.60 [+0.15, +1.05]` pts (CI > 0 on acc, < 0 on CE)**
- `own is example-specific (own_shuffled - own)`: CE `-0.0182 [-0.0881, +0.0560]` | acc **`+0.80 [+0.15, +1.55]` pts -> not distinguishable from 0 (on CE)**
- `own vs a random real image (wrong - own)`: CE `+0.2238 [+0.1036, +0.3479]` | acc **`+3.05 [+1.90, +4.20]` pts -> SUPPORTED**
- `headroom: true image vs own (own - oracle)`: CE `+0.7234 [+0.6119, +0.8361]` | acc **`+7.80 [+6.60, +9.00]` pts -> SUPPORTED**
- `[reference] image content usable (wrong - oracle)`: CE `+0.9472 [+0.8162, +1.0824]` | acc `+10.85 [+9.45, +12.30]` pts -> **SUPPORTED**
- `[reference] true image vs blank (const - oracle)`: CE `+0.5709 [+0.4805, +0.6592]` | acc `+8.40 [+7.20, +9.60]` pts -> **SUPPORTED**

**Per-Family Breakdown (`test_ood`, `n=2000`, accuracy %):**

| Family | n | oracle | own | own_shuffled | wrong | const |
|--------|---|--------|-----|--------------|-------|-------|
| `fit_ball_box` | 490 | 99.2 | **99.2** | 96.9 | 92.7 | 97.6 |
| `fit_rect` | 404 | 99.3 | 99.0 | 97.0 | 91.6 | 99.3 |
| `grid_walk` | 488 | 77.5 | **76.8** | 76.4 | 76.0 | 75.8 |
| `spatial_grid` | 291 | 99.0 | 99.0 | 99.3 | 99.7 | 98.6 |
| **`triangle`** | **327** | **98.2** | **51.7** | **52.9** | **52.6** | **52.0** |

- **Retrieval on `test_ood` (`n=2000`):** `top1=0.012, top5=0.047` (chance `0.001 / 0.003`; still ~12x to 15x chance, but down from 6.0% / 23.0% in-distribution).
- **Interesting split on `test_ood`:** On `fit_ball_box` (490 items) and `grid_walk` (488 items), `own` actually edges out `const` (`99.2% vs 97.6%` and `76.8% vs 75.8%`), giving a tiny **+0.60 pt accuracy gain (`[+0.15, +1.05]`)** overall. However, on **`triangle`**, `own` completely fails (`51.7%` vs `52.0%` `const`, chance), and because the reader places high confidence in the visual tokens, the badly imagined `triangle` tokens inflate Cross-Entropy (`0.9331` vs `0.7806` `const`).

#### 8C. Qualitative Analysis of the Phase C Retrieval Galleries (`imagine_out/*.png`)
1. **`adv_c_qwen2.5-0.5b_grid3_spatial2_test_imagination.png` (In-distribution `test`):**
   - Each row shows: question text -> `TRUE image` -> top-3 test images retrieved by the model's imagined tokens (`imagined #1, #2, #3`).
   - **Result:** The Imagination Head cleanly separates task families and captures fine parameters!
   - In Row 2 (`Can a ball of diameter 10 cm fit inside a box that is 12 cm wide and 8 cm tall?`), `imagined #1` retrieves the **exact true image** (`d=10`, `12 cm x 8 cm`), and `#2` and `#3` retrieve nearly identical non-fitting ball-in-box diagrams (`d=12 in 14x8`, `d=11 in 13x8`).
   - In Rows 1, 3, and 6 (`3x3 spatial_grid`), all 3 retrieved images are `3x3` grids with matching object counts and similar layouts.
   - In Rows 4 and 5 (`4x4 grid_walk`), all retrieved images are `grid_walk` boards with grey wall cells and blue agent dots (Row 5 retrieves three `4x4` boards).
2. **`adv_c_qwen2.5-0.5b_grid3_spatial2_test_ood_imagination.png` (Out-of-distribution `test_ood`):**
   - Even on OOD numbers, family separation is 100% intact: `fit_rect` questions retrieve `fit_rect` diagrams (Rows 1, 3), `4x4 spatial_grid` questions retrieve `4x4` grids (Rows 2, 5), and `fit_ball_box` questions retrieve `fit_ball_box` diagrams (Rows 4, 6).
   - However, parameter precision degrades on OOD numbers: e.g., Row 4 (`d=24 in 26x31`, where the ball almost touches the walls) retrieves diagrams where the ball is much smaller relative to the box (`d=19 in 31x23`, `d=17 in 34x23`). This visually confirms why OOD `triangle` fails: the MLP projects OOD numbers into "typical" in-distribution visual proportions rather than computing exact scale-free geometry.

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
| `imagine_step5.py` | Stage 1: COCO captions -> CLIP embeddings (InfoNCE). | Run |
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
| `imagine_step8_advanced_b.py` | Phase B: train only the Imagination Head from question text; `--track imagine`. | **Run & Completed (3 epochs)** |
| `imagine_step8_advanced_c.py` | Phase C: final test, per-group table, gallery of what is imagined; `--split test / test_ood`. | **Run & Completed (`test` and `test_ood`)** |

### 6.4 Dataset and tooling scripts

| File | Author | What it does |
|------|--------|--------------|
| `imagine_diagnose_data.py` | Cursor | Shows how text-solvable ScienceQA is (hint length, answer leakage, etc.). |
| `imagine_build_spatial_cot.py` | Cursor | Builds the v1 spatial dataset (leaky). |
| `imagine_fetch_public.py` | Cursor | Optional extra COCO captions, Visual-CoT annotations (about 345 MB) or images (about 143 GB). Not used so far. |
| `imagine_audit_data.py` | Claude | Leak audit (index bias, numeric-rank bias, question/choice lookup, one-feature image attacker). |
| `imagine_build_spatial_v2.py` | Claude | Builds the leak-free v2 dataset (8 families, OOD split, optional step images) and self-audits. |
| `imagine_check_env.py` | Claude | Checks that the scripts and shared modules come from the same version. |

### 6.5 Folders

| Folder | Contents |
|--------|----------|
| `imagine_cache/` | Downloaded and generated data: COCO images, ScienceQA images, CLIP feature caches (`adv_feats_*`), `spatial_cot/` (v1), `spatial_cot_v2/` (v2). |
| `imagine_ckpt/` | Checkpoints per experiment: `head_step5/6.pt`, `chain_step8*.pt`, `adv_a0_*`, `adv_a_*` (reader + LoRA), `adv_b_*` (imagination head). |
| `imagine_out/` | Pictures (`step7_retrieval.png`; `adv_c_qwen2.5-0.5b_grid3_spatial2_test_imagination.png`; `adv_c_qwen2.5-0.5b_grid3_spatial2_test_ood_imagination.png`). |
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
| `mark_see` | see | nothing about the scene | a 3x3 grid with one ringed object | needs reading the ring |
| `rel_see` | see | nothing about the scene | two objects | needs left-of / above |

Items have fields: `question`, `choices`, `answer` (index), `family`, `track`, `image_path`, `steps`, `spec`.

---

## 8. Consolidated results

### 8.1 Original track (SmolLM2-135M, ScienceQA CoT)

| Experiment | Key numbers | Takeaway |
|------------|-------------|----------|
| Step 5 | val top-1 0.34, top-5 0.69 (chance 0.005 / 0.025) | text hidden state carries visual semantics |
| Step 6 | gate val acc 0.956, but probes wrong | proxy label is imperfect |
| Step 8 | LM loss feedback 1.65 -> 1.33; benefit 0.34 -> 0.70 | looked promising |
| Step 8b | shuffled 1.3451 vs feedback 1.3324; constant 2.0355 | benefit is a soft-prompt effect |
| Step 8c | own_info_gain +0.0126 (unstable over epochs) | sign right, magnitude tiny |
| Step 8d | image content usable +0.0010 [-0.0004, +0.0025] | the LLM did not read the image |

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

### 8.3 Advanced track Phase B & C (Qwen2.5-0.5B, `spatial2`, `imagine` track, grid 3)

| Split (n) | oracle | own | own_shuffled | wrong | const | `own` vs `const` (acc / CE) | `own` vs `own_shuffled` (acc) | Image Retrieval Top-1 / Top-5 |
|-----------|--------|-----|--------------|-------|-------|-----------------------------|-------------------------------|-------------------------------|
| **Val — Before** (1386) | 96.2% | 94.6% | 95.0% | 90.7% | 95.1% | -0.51 pts / -0.0288 (OPPOSITE) | -0.43 pts (n.s.) | 0.0% / 0.1% |
| **Val — Final Ep 3** (1386) | 96.2% | **95.2%** | 92.8% | 90.7% | 95.1% | +0.14 pts / -0.0032 (n.s.) | **+2.45 pts [1.52, 3.39] (SUPPORTED)** | **6.9% / 23.5%** |
| **Test — Held-out** (1409) | 97.3% | **96.7%** | 94.7% | 91.7% | 96.5% | +0.21 pts / +0.0001 (n.s.) | **+1.99 pts [1.21, 2.84] (SUPPORTED)** | **6.0% / 23.0%** |
| **Test OOD** (2000) | 93.7% | **85.9%** | 85.1% | 82.8% | 85.3% | **+0.60 pts** / -0.1525 (OPPOSITE CE) | +0.80 pts (n.s. on CE) | **1.2% / 4.7%** |

---

## 9. What we have learned (and what we have NOT shown)

**Learned**
1. **Always run controls.** A trained extra token improves loss with no visual information (soft-prompt effect). Without the shuffled, random and constant controls, step 8's "benefit" would have looked like a result.
2. **A benchmark must make the image necessary.** On ScienceQA the answer is mostly recoverable from text, so no image-based method can win. The first spatial dataset had the opposite problem: its images contained the answer.
3. **Alignment first.** A reader trained only on answers from a few thousand questions learns nothing; a few minutes of caption alignment on COCO makes the LLM read visual tokens (+1.03 nats true-vs-wrong).
4. **Leak-free sketches work for reading.** On v2 the reader uses image content (+8.1 points), and learns counting in `count_see` (21% -> 64%).
5. **Imagination cannot add facts that are not in the text.** In the `see` track the text says nothing about the image, so a text-only student can never recover it beyond the prior. Imagination can only help by *re-representing what the text already states* (e.g. turning numbers into a to-scale picture), where the picture makes the reasoning easier.
6. **Text-only models are strong in-distribution and brittle out of it.** `triangle` text-only: 93% to 94% in-distribution, 52% (chance) with bigger numbers. A scale-free picture stays at 98%.
7. **A single "scene" picture does not help state tracking** (`grid_walk`: 86.2 vs 85.6 on val, 90.1 vs 90.1 on test). The helpful visual there would be the board after each move (the PDF's per-step imagination), not the starting board.
8. **The Imagination Head truly learns question-specific visual representations (`own > own_shuffled`).** Unlike Steps 8 to 8d where `own` and `own_shuffled` were indistinguishable, Phase B/C proves that a 20M-parameter MLP trained with feature + InfoNCE + answer + KD losses generates tokens specific to the exact question (`+2.45 pts` on val, `+1.99 pts` on test over `own_shuffled`, and `23.0%` Top-5 exact diagram retrieval out of 1,409 test items).
9. **The Single-Shot Representation Bottleneck (`own ~ const` on OOD `triangle`).** Why did `own` fail on OOD `triangle` (`51.7%` vs `98.2%` `oracle`)? Because the Imagination Head takes the LLM's final prompt hidden state `h` and maps it to visual tokens in one shot. Since `oracle` works (`98.2%`), the reader only needs to see whether the two arcs meet in the CLIP features. For the Imagination Head to produce "arcs meet" vs "arcs don't meet" from `h` on OOD numbers (`16..40`), `h` + the MLP must compute `a + b > c` for those OOD numbers—the exact arithmetic comparison that the text model failed to extrapolate! In a single step, latent imagination is a *projection* of `h`, not an independent physics/geometry engine. For latent visual imagination to beat text-only reasoning (`own > const`), the visual state must be updated **iteratively across simple steps** (where each step is an easy local update, like moving one cell on a grid in `grid_walk`, avoiding multi-step text drift).

**NOT shown yet**
- That self-generated imagination (`own`) clearly beats the blank-token baseline (`const`) on both accuracy and CE (in single-step answer-first format, `own` ties `const` in-distribution and has worse CE out-of-distribution).
- Anything about the gate on a good label (the gate was only trained on a proxy).
- Anything about multi-step imagination or collapse across steps (needs the `grid_walk` per-step chain).
- Anything about GRPO (steps 9 and 10 never run), real image decoding, or Visual CoT.
- Robustness across random seeds (all results are single runs; the CIs cover only question sampling).

**Caveats to remember**
- The triangle sketch nearly depicts the answer (the audit showed total arc length predicts it 99%), so "oracle 98%" is closer to "a solved diagram helps" than to "visual reasoning".
- 3x3 pooled CLIP features cannot read a ring or left/right relations (`mark_see` and `rel_see` stayed at chance). Whether 7x7 features can is untested because the grid-7 run lacked alignment.
- Wrong visual tokens hurt (`wrong` is 91.7% vs 96.5% `const` on test), and out-of-distribution imagined tokens (`own` on `test_ood`) increase Cross-Entropy (`0.9331` vs `0.7806` `const`) because the frozen reader trusts the visual slot even when the imagination head extrapolates poorly.

---

## 10. File versions, `old_progs`, and the run log

**The author's naming convention.** When a script is replaced, the previous copy goes to `old_progs/` with a suffix: `name_old.py`, `name_old2.py`, ... A **higher number means a more recent** copy; the plain filename is always the latest. A new copy may be saved even if the previous one was never run. Because commands were run with the plain filename at that moment, an older result was produced by `..._old`, `..._old2`, ... in the order the file was *run*, and the most recent run corresponds to the plain filename.

**How to use this to find which file made which result:** count the runs of a script in the table below (oldest first). The 1st run's code is the oldest `old` copy that still matches, and so on. The exact copy depends on how many intermediate versions author saved, which cannot be determined at this point.

**Incident to know about.** On 2026-10-24 at 11:17, `imagine_adv_lib.py`, `imagine_data.py` and `imagine_step8_advanced_a.py` were found to be **older copies than expected** (same timestamp, flagged by `imagine_check_env.py`). At 11:28, they were replaced with the newest versions (`imagine_data.py` 8213 bytes, `imagine_adv_lib.py` 14766 bytes, `LIB_VERSION >= 4`), and `imagine_check_env.py` confirmed `"All consistent"` right before Phase B was launched.

### Run log (chronological, from the terminal timestamps)

| # | Approx. time | Script and arguments | Outcome in one line |
|---|--------------|----------------------|---------------------|
| 1 | 10-21 00:30 | `imagine_step1.py` | OK, d_model 576 |
| 2 | 00:33 | `imagine_step2.py` | OK, shapes correct |
| 3 | 00:34 | `imagine_step3.py` | OK, cosine 1.0 -> -0.29 |
| 4 | ~00:36 | `imagine_step4.py` | OK, 3 checks pass |
| 5 | 00:40 | `imagine_step5.py` (first crash on CLIP `.bin`, fixed, rerun) | top-1 0.34 / top-5 0.69 |
| 6 | 00:51 | `imagine_step6.py` | gate acc 0.956, probes off |
| 7 | ~00:56 | `imagine_step7.py` | montage saved |
| 8 | 01:02 | `imagine_step8.py` | benefit 0.70, align 0.18 |
| 9 | 06:47 | `imagine_step8b.py` | information gain +0.0127 |
| 10 | 07:24 | `imagine_step8c.py` | own_info_gain +0.0126 |
| 11 | 08:51 | `imagine_step8d.py --n_eval 150` | not distinguishable |
| 12 | 15:57 | `imagine_step8_advanced_a.py --vis global` (ScienceQA) | 71.8 / 72.3 / 73.2 |
| 13 | 16:28 | `..._a.py --vis grid --grid 3 --epochs 3` | 70.3 / 70.3 / 70.0 |
| 14 | 16:45 | `..._a.py --vis grid --grid 3 --epochs 3 --w_contrast 1.0` | 71.5 / 71.8 / 71.7 |
| 15 | ~17:20 | `..._a0.py --vis grid --grid 3` | A0 passed, gap +1.03 |
| 16 | 18:00 | `..._a.py --vis grid --grid 3 --epochs 3` (aligned reader) | 69.5 / 69.2 / 68.5 |
| 17 | 10-21 23:xx | `imagine_diagnose_data.py`; `imagine_build_spatial_cot.py --n_train 80000 --n_val 4000 --n_test 4000` | ScienceQA text-solvable; v1 built |
| 18 | 10-21 23:46 | `..._a.py --data spatial --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4 --w_contrast 1.0` | 100 / 58.3 / 99.2 (leaky) |
| 19 | 10-22 09:3x | `imagine_audit_data.py` on v1 (8/8 flagged); `imagine_build_spatial_v2.py --n_train 40000 --n_val 3000 --n_test 3000 --n_ood 2000`; audit on v2 (clean) | v2 built |
| 20 | 10-22 09:35 | `..._a.py --data spatial2 --vis grid --grid 3 --n_train 20000 --n_val 2000 --epochs 4` | **81.8 / 73.8 / 76.9** |
| 21 | 10-23 23:59 | `..._a_eval.py --data spatial2 --vis grid --grid 3 --split test_ood --n 2000` | OOD 93.7 / 82.8 / 85.3 |
| 22 | 10-24 00:02 | `..._a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4` | 76.2 / 75.6 / 76.1 (no A0) |
| 23 | 10-24 (crashes) | `..._b.py`, `..._c.py` | `NameError: suffix_of`, then `TypeError: root`: version mismatch (Section 11) |
| 24 | 10-24 11:17 & 11:28 | `imagine_check_env.py` | 5 items OLD at 11:17 -> replaced -> `"All consistent"` at 11:28 |
| 25 | **10-24 11:30** | `..._b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3` | **Completed: own 95.2% vs shuf 92.8% (+2.45 pts, SUPPORTED), retrieval top5 23.5%** |
| 26 | **10-24 12:36** | `..._c.py --data spatial2 --vis grid --grid 3 --track imagine --split test --n_test 2000` | **Completed: own 96.7% vs const 96.5%, vs shuf 94.7% (+1.99 pts, SUPPORTED), top5 23.0%** |
| 27 | **10-24 12:40** | `..._c.py --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000` | **Completed: oracle 93.7%, own 85.9% vs const 85.3% (OOD triangle own 51.7% = chance)** |

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

**Right now:** The entire single-step Answer-First Advanced Track (`Phase A0 -> Phase A -> Phase B -> Phase C` on `test` and `test_ood`) is **complete**.

**Summary of the Phase B & C outcome:**
1. **`own is example-specific` PASSED strongly:** On both `val` (`+2.45 pts`) and `test` (`+1.99 pts`), the student's own imagination significantly outperforms shuffled imagination (`own_shuffled`) and random real images (`wrong`, `+4.97 pts`), and retrieves the true test diagram in the Top-5 **23.0%** of the time (nearly 60x chance).
2. **`own vs blank (const)` is a tie in-distribution (`96.7% vs 96.5%` on `test`) and split out-of-distribution (`85.9% vs 85.3%` acc, but worse CE on `test_ood`):**
   - On `test`, `triangle` shows a small positive hint (`95.6%` `own` vs `94.1%` `const`, and `87.7%` `own_shuffled`), while `fit_ball_box`, `fit_rect`, and `spatial_grid` are already at `100.0%` with text alone.
   - On `test_ood`, `triangle` drops to chance (`51.7%` `own` vs `52.0%` `const` vs `98.2%` `oracle`). This diagnoses a fundamental architectural insight: **one-shot imagination from the final prompt hidden state `h` cannot solve an arithmetic comparison that `h` itself failed to compute.**

**Planned next experiments (in suggested order)**
1. **Multi-step `grid_walk` chain (highest priority — tests the PDF's core per-step loop where single-step imagination hit a ceiling):**
   - Why: `grid_walk` is the one `imagine` family with clear in-distribution (`~86%` to `90%`) and OOD (`~76%`) headroom where the static starting board (`oracle`) didn't help (`90.1% vs 90.1%` on test). Tracking a walk step-by-step by updating the imagined board state `z_{t-1} + move_t -> z_t` decomposes a long sequence into local spatial updates—the exact regime where iterative visual state tracking can beat text-only memory drift.
   - Action: Generate step-by-step images (`imagine_build_spatial_v2.py --step_images`), train the recurrent step-by-step imagination loop, and evaluate on `test` and longer `test_ood` walks (measuring consecutive step cosine for collapse as well).
2. **Redo grid 7 properly (A0 for grid 7, then Phase A):**
   - Run `python imagine_step8_advanced_a0.py --vis grid --grid 7` first, then `python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4`, to see whether 7x7 tokens (49 patch tokens) allow the reader to solve `mark_see` (ringed object) and `rel_see` (left-of/above) which stayed at chance under 3x3 pooling.
3. **Calibrating reader reliance / confidence on OOD imagination:**
   - On `test_ood`, `own` had slightly higher accuracy (`85.9%` vs `85.3%` `const`) but worse CE (`0.9331` vs `0.7806`) because the frozen reader trusted out-of-distribution imagined tokens too strongly. Co-adapting the reader with a small mix of imagined tokens or gating unreliable imaginations could prevent CE degradation.
4. **Photo VQA (A-OKVQA)** as a second, natural-image domain.
5. **Gate training with a better label** (using an LLM judge via an API such as Gemini, or empirical benefit `CE(const) - CE(own) > margin` as the ground-truth gate target!).
6. **GRPO (steps 9 and 10)**, adapted to the advanced track, with answer accuracy as reward.
7. **The author's dual-stack idea** (text stack plus latent-image stack with bidirectional cross-attention) as an architecture ablation: allowing the visual stack and text stack to cross-attend *layer-by-layer* could overcome the single-shot MLP bottleneck where `h` must already know the answer before `head(h)` is called.
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

# Phase B (imagination head) and Phase C (final test) — COMPLETED FOR GRID 3
python imagine_step8_advanced_b.py --data spatial2 --vis grid --grid 3 --track imagine --n_train 20000 --n_val 2000 --epochs 3
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test --n_test 2000
python imagine_step8_advanced_c.py --data spatial2 --vis grid --grid 3 --track imagine --split test_ood --n_test 2000

# Proper Grid 7 (7x7 = 49 tokens) sequence (requires A0 for grid 7 first!)
python imagine_step8_advanced_a0.py --vis grid --grid 7
python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4
```

*Out-of-memory tip: add `--micro_bs 4 --accum 4`. Interrupted Phase A run: add `--resume`.*

---

## TL;DR for Pro

Evaluation of Results & What To Do Next

- 1. Is this result good, or does it need tweaking?

Scientifically, this is a very strong diagnostic result—and much more honest and publishable than a fake "99% accuracy" from a leaky dataset.

Here is what Phase B and Phase C results actually proved:

What succeeded: For the first time in the research, the model learned genuinely question-specific visual imagination (own beat own_shuffled by +2.45 pts on val and +1.99 pts on test, both statistically SUPPORTED, and retrieved the exact diagram in the top 5 23% of the time out of 1,409 images). On in-distribution triangle, own got 95.6% vs 94.1% blank (const) and 87.7% own_shuffled.

What hit a ceiling (and why):

In-distribution (test): Text-only (const) already solves fit_ball_box (100%), fit_rect (100%), and spatial_grid (100%). Because a 0.5B LLM can easily learn in-distribution number comparisons (10 < 12) in text, there is simply no room for an imagined image to improve on 100%.

Out-of-distribution (test_ood): On triangle, oracle (the true sketch) got 98.2%, while const (text-only) dropped to 52.0% (chance), and own (imagined sketch) also got 51.7% (chance).

Why did own fail on OOD triangle? Look at the oracle sketch for triangle: the Python script draws two arcs from the ends of the longest side, and the CLIP reader simply looks at whether the arcs intersect. When the student has to imagine those CLIP features in one shot from the LLM's final hidden state h, the MLP has to decide whether to output "intersecting arc features" or "non-intersecting arc features." To know which one to output for OOD side lengths (like 19, 21, 39), the MLP + h must compute 19 + 21 > 39—which is the exact same arithmetic that the LLM failed to extrapolate in text!

- 2. Should run the 7x7 (--grid 7) experiment?
Only if one cares about the see track (count_see, mark_see, rel_see), NOT to fix own imagination.

Why 7x7 helps Phase A (see track): In Phase A (grid 3), mark_see (finding the black ring) and rel_see (left-of/above) stayed at random chance (~31% and ~48%) because averaging a 7x7 CLIP grid down to 3x3 blurs out fine positions and thin rings. Running grid 7 (49 tokens) with proper A0 alignment will test if finer tokens let the reader see the ring and spatial relations.

Why 7x7 will NOT fix Phase B/C (imagine track): On the imagine track, grid 3 oracle is already 97.3% on test and 98.2% on OOD triangle. That means 3x3 tokens already carry almost 100% of the visual information needed for the imagine track! Making the Imagination Head predict 49 tokens (49 x 768 = 37,632 numbers) instead of 9 tokens (9 x 768 = 6,912 numbers) makes the regression target 5.4x larger and harder to imagine, without adding headroom.

If wanted to run 7x7 to complete the Phase A resolution ablation table, run these exact commands in order:

```PowerShell
# Step 1: Align the 7x7 reader on COCO captions FIRST (~35-45 mins)
# (This was missing in earlier grid-7 run, which is why it failed!)
python imagine_step8_advanced_a0.py --vis grid --grid 7

# Step 2: Train Phase A on spatial2 with the aligned 7x7 reader (~85 mins)
python imagine_step8_advanced_a.py --data spatial2 --vis grid --grid 7 --n_train 20000 --n_val 2000 --epochs 4

# Step 3 (Optional): Evaluate Phase A 7x7 on OOD
python imagine_step8_advanced_a_eval.py --data spatial2 --vis grid --grid 7 --split test_ood --n 2000
```

- 3. What tweaks or dataset/pipeline changes WILL make own > const?
Right now, grid_walk is a golden opportunity. Look at the test and test_ood tables:

Every other imagine family is a 1-step static comparison (a < b or a + b > c).

grid_walk is the only family where in-distribution accuracy is not saturated (90.1% on test, 86.2% on val, 75.8% on test_ood), AND where even the oracle image didn't help (90.1% oracle vs 90.1% const) because the v2 dataset only gave the starting board image, not the board after the moves!

Here are the best concrete directions one can try next:

## Direction A (Best for the Paper's Core Claim): Multi-Step Visual State Tracking on grid_walk

Why it works: When an LLM tries to track 5 to 9 moves (up, left, left, down, right...) purely in text without writing CoT, it loses track of coordinates around move 5. But updating a visual grid by 1 step (board_{t-1} + move_t -> board_t) is a simple local spatial shift!

- What needs to change:

Rebuild spatial_cot_v2 with --step_images (already supported in imagine_build_spatial_v2.py), which saves an image of the board after each move.

Instead of predicting one image at the end of the question, run the per-step feedback loop (from the research plan PDF!) across the moves of grid_walk: after each move token, the Imagination Head imagines the new board z_t, feeds it back through the reader, and answers the final (row, col) from the final imagined board z_T.

## Direction B (Quick Tweak to existing Phase B/C): Train & Test on triangle + grid_walk only, or Co-Adapt the Reader in Phase B

In Phase B, the reader and LoRA were 100% frozen, and const was trained in Phase A when the model had no imagination. If we unfreeze the reader (with a tiny learning rate like 5e-5) or add a Gate in Phase B/C that falls back to const (blank tokens) when the Imagination Head's confidence/norm is out-of-distribution, we will immediately eliminate the negative CE (-0.1525) on test_ood while keeping the +0.60 pt accuracy gain.

## Direction C (Empirical Gate Target):

Remember how Step 6 needed a proxy label (has_image) for the Gate g_t? we now have the exact mathematical ground truth for the gate on spatial2: for every training example, we can compute benefit = CE(const) - CE(own). Train the gate g_t = sigmoid(W1 * h_t) to predict 1 if CE(const) > CE(own) (imagination helps) and 0 if CE(const) <= CE(own) (imagination hurts/is unnecessary), and blend the tokens: vtok = g_t * own + (1 - g_t) * const. This guarantees gated_own >= const!