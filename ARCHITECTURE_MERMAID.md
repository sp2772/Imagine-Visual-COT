# Imagination-Head Project: Architecture and Pipeline Diagrams

All diagrams are Mermaid flowcharts. To render them: open this file in VS Code with a Markdown preview that supports Mermaid, or paste any code block into https://mermaid.live , or view it on GitHub.

Colour legend used in every diagram: **green** = done and run, **blue** = idea or design, **orange** = key finding, **grey dashed** = planned or never run, **red** = failure or flawed result.

| # | Diagram | What it shows |
|---|---------|---------------|
| 1 | Project journey | Every phase in order, with the decision each result caused |
| 2 | Original PDF architecture | The design in `research_plan_new.pdf` |
| 3 | Steps 1 to 7 | The basic modules (SmolLM2-135M) |
| 4 | Steps 8 to 8d | Joint training and the control experiments |
| 5 | Advanced track pipeline | A0, A, B, C with the data and checkpoints |
| 6 | Core architecture (answer-first) | What actually runs at test time and in training |
| 7 | Phase B losses | How the Imagination Head is trained |
| 8 | Evaluation conditions | What oracle / wrong / const / own / own_shuffled feed in |
| 9 | Datasets and leak audit | ScienceQA, spatial v1 (leaky), spatial v2 (clean) |
| 10 | Planned: multi-step grid_walk | The next experiment (per-step imagination) |
| 11 | Future idea: dual-stack | The two-stack cross-attention idea |
| 12 | Programs and checkpoints map | Which script produces and consumes what |

---

## 1. Project journey (all phases and decisions)

```mermaid
flowchart TD
    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef idea fill:#dbe9ff,stroke:#1565c0,color:#000
    classDef finding fill:#ffe0b2,stroke:#e65100,color:#000
    classDef bad fill:#ffcdd2,stroke:#c62828,color:#000
    classDef planned fill:#eeeeee,stroke:#757575,stroke-dasharray: 5 5,color:#000

    PDF["research_plan_new.pdf<br/>Imagination Head + Gate + feedback loop + anti-collapse<br/>4 training stages, LoRA bf16"]:::idea

    subgraph T1["Track 1 - SmolLM2-135M, ScienceQA chain-of-thought (steps 1 to 8d)"]
        direction TB
        S17["Steps 1-7: modules built and sanity-checked<br/>backbone, head, gate, feedback chain,<br/>COCO alignment, gate, retrieval"]:::done
        S8["Step 8: joint training, LLM frozen<br/>LM loss 1.65 to 1.33, benefit 0.34 to 0.70"]:::done
        S8b["Step 8b: controls<br/>feedback 1.3324, shuffled 1.3451,<br/>random 1.5630, constant 2.0355"]:::done
        S8c["Step 8c: teacher forcing + LoRA, k=4 tokens<br/>own_info_gain +0.0126, unstable across epochs"]:::done
        S8d["Step 8d: 150 fresh questions + bootstrap CI<br/>image content usable +0.0010, not distinguishable"]:::done
        F1{{"Finding 1: the gain was a soft-prompt effect<br/>the text-only LLM never read image content"}}:::finding
        S17 --> S8 --> S8b --> S8c --> S8d --> F1
    end

    subgraph T2["Track 2 - Qwen2.5-0.5B, answer-first, reader (CLIP to LLM tokens)"]
        direction TB
        A0["A0: caption alignment of the reader on 20,000 COCO images<br/>true 2.447 vs wrong 3.476 vs blank 2.939 - PASSED"]:::done
        ASQ["Phase A on ScienceQA (4 runs)<br/>global, grid3, grid3+contrast, grid3 after A0<br/>oracle = wrong = blank, about 70% - NOT PASSED"]:::bad
        DIAG["imagine_diagnose_data.py<br/>65% have hints, 14% contain the gold string,<br/>26% knowledge questions"]:::done
        F2{{"Finding 2: ScienceQA is text-solvable<br/>swapping the image changes nothing"}}:::finding
        V1["Spatial v1 built by Cursor agent<br/>80,000 items, 8 families"]:::done
        AV1["Phase A on v1: oracle 100.0, wrong 58.3, blank 99.2<br/>looks like a pass"]:::bad
        AUD1["imagine_audit_data.py on v1<br/>8 of 8 families flagged"]:::done
        F3{{"Finding 3: v1 leaks the answer<br/>verdict text and green/red colour in images,<br/>answer index 0, choice sets encode the answer"}}:::finding
        V2["Spatial v2 built, leak-free<br/>imagine track + see track + test_ood"]:::done
        AUD2["Audit on v2: 0 of 8 flagged"]:::done
        AV2["Phase A on v2: oracle 81.8, wrong 73.8, blank 76.9<br/>oracle vs wrong +8.10 pts, SUPPORTED"]:::done
        EV["Eval on test_ood<br/>triangle: oracle 98.2, blank 52.0 (chance)"]:::done
        G7["Grid 7 run without A0 alignment<br/>inconclusive, confounded"]:::bad
        PB["Phase B: train only the Imagination Head<br/>20.2 M params, 3 epochs, about 17 min each"]:::done
        PC["Phase C: held-out test and test_ood"]:::done
        F4{{"Finding 4: own imagination is example-specific<br/>+2.45 pts val, +1.99 pts test over own_shuffled<br/>but own ties blank, and fails on OOD triangle (51.7 vs 52.0)"}}:::finding
        A0 --> ASQ --> DIAG --> F2 --> V1 --> AV1 --> AUD1 --> F3 --> V2 --> AUD2 --> AV2 --> EV --> G7 --> PB --> PC --> F4
    end

    NEXT["Planned next experiments"]:::planned
    N1["Multi-step grid_walk chain<br/>per-step imagined board state (the PDF loop)"]:::planned
    N2["Grid 7: run A0 for grid 7, then Phase A"]:::planned
    N3["Gate with empirical target<br/>CE(const) minus CE(own), blend tokens"]:::planned
    N4["Co-adapt the reader, photo VQA (A-OKVQA),<br/>GRPO steps 9-10, dual-stack ablation"]:::planned

    PDF --> T1
    F1 -->|"decision: bigger model, answer-first,<br/>aligned reader, controls with CIs"| T2
    F4 --> NEXT
    NEXT -.-> N1
    NEXT -.-> N2
    NEXT -.-> N3
    NEXT -.-> N4
```

---

## 2. Original architecture from `research_plan_new.pdf`

```mermaid
flowchart TD
    classDef idea fill:#dbe9ff,stroke:#1565c0,color:#000
    classDef out fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef loss fill:#fff3c4,stroke:#f9a825,color:#000

    IN["Input at reasoning step t<br/>text prompt and optional image prompt"]:::idea
    LLM["LLM backbone, e.g. LLaMA<br/>produces hidden state h_t"]:::idea
    IN --> LLM

    LLM --> NTH["Next-token head"]:::idea
    NTH --> WT["text token w_t"]:::idea
    WT --> FOUT["Final text output<br/>shown to the user"]:::out

    LLM --> IH["Causally-Enforced Imagination Head<br/>z_t = W2 x h_t"]:::idea
    IH --> Z["Visual vector z_t<br/>the imagination latent"]:::idea
    LLM --> GT["Gating mechanism<br/>g_t = sigmoid of W1 x h_t, value 0 to 1"]:::idea

    Z --> DEC{"g_t over threshold?"}
    GT --> DEC
    DEC -->|"yes: visual question"| DIMG["Image decoder<br/>visible output"]:::out
    DEC -->|"no: abstract question"| HID["Imagination stays hidden<br/>but still guides the text"]:::idea

    Z --> CM["Contrastive module<br/>keeps z_t different from z_t-1"]:::loss
    CM -.->|"feedback for N steps:<br/>z_t-1 fed back into the LLM"| LLM

    subgraph TRAIN["Training plan in the PDF"]
        direction LR
        ST1["Stage 1: COCO captions<br/>text to image-embedding space"] --> ST2["Stage 2: Visual CoT<br/>region grounding"] --> ST3["Stage 3: ScienceQA + modified GRPO<br/>accuracy and format rewards"] --> ST4["Stage 4: joint multi-task training<br/>text + imagination + gate losses"]
    end
```

---

## 3. Steps 1 to 7: building the basic modules (SmolLM2-135M)

```mermaid
flowchart LR
    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef ckpt fill:#f3e5f5,stroke:#6a1b9a,color:#000

    subgraph DATA["Data and helpers (imagine_data.py)"]
        direction TB
        COCO["COCO captions<br/>lmms-lab/COCO-Caption<br/>2,000 images"]:::data
        SQA["ScienceQA<br/>400 image + 400 no-image items"]:::data
        CLIP["CLIP ViT-B/32<br/>512-d image and text embeddings"]:::data
    end

    S1["Step 1: load SmolLM2-135M<br/>extract h_t, d_model 576<br/>hidden shape 1 x 28 x 576"]:::done
    S2["Step 2: ImaginationHead<br/>LayerNorm, W2 to 512-d z, W1 to gate<br/>297,153 params, z norm 1.0"]:::done
    S3["Step 3: anti-collapse loss demo<br/>consecutive cosine 1.000 to -0.286"]:::done
    S4["Step 4: FeedbackChain<br/>context I-1, seg0, I0, seg1, ...<br/>3 checks: feedback matters, causal, gradients flow"]:::done
    S5["Step 5: Stage 1 COCO alignment<br/>InfoNCE, text hidden state to CLIP image<br/>val top1 0.34, top5 0.69 (chance 0.005 / 0.025)"]:::done
    S6["Step 6: gate training<br/>proxy label: question has an image<br/>val acc 0.956, probes partly wrong"]:::done
    S7["Step 7: retrieval decoder stand-in<br/>nearest COCO image to z<br/>montage in imagine_out"]:::done

    C5[("head_step5.pt")]:::ckpt
    C6[("head_step6.pt")]:::ckpt

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    COCO --> S5
    CLIP --> S5
    S5 --> C5 --> S6
    SQA --> S6
    S6 --> C6 --> S7
    COCO --> S7
```

---

## 4. Steps 8 to 8d: joint training and the control experiments

```mermaid
flowchart TD
    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef finding fill:#ffe0b2,stroke:#e65100,color:#000
    classDef loss fill:#fff3c4,stroke:#f9a825,color:#000

    D8["ScienceQA solutions split into reasoning steps<br/>prompt, Step 1..4, Answer<br/>CLIP text embedding of every step,<br/>CLIP image embedding of the question image"]:::done

    subgraph S8["Step 8: LLM frozen, trainable: head, gate, projector, img_start"]
        direction TB
        CH["FeedbackChain over the steps<br/>I t-1 = proj(z t-1) injected before each step"]:::done
        L1["LM loss on all step text"]:::loss
        L2["Alignment: z_t vs CLIP text of step t<br/>InfoNCE across the batch"]:::loss
        L3["Image loss: z vs CLIP image"]:::loss
        L4["Diversity loss: consecutive cosine below 0.8"]:::loss
        L5["Gate BCE: has image"]:::loss
        CH --> L1
        CH --> L2
        CH --> L3
        CH --> L4
        CH --> L5
    end
    R8["Result after 2 epochs<br/>LM loss 1.65 to 1.33, align top1 0.035 to 0.182,<br/>gate acc 0.775 to 0.943, consecutive cos 0.861 to 0.787"]:::done

    subgraph S8B["Step 8b: control, same checkpoint, 80 val questions"]
        direction LR
        C1["feedback<br/>own z fed back: 1.3324"]:::done
        C2["shuffled<br/>another example z: 1.3451"]:::done
        C3["random_z<br/>random unit vectors: 1.5630"]:::done
        C4["constant<br/>same token every step: 2.0355"]:::done
    end
    F8B{{"information gain = shuffled - feedback = +0.0127<br/>soft-prompt effect = constant - shuffled = +0.6905<br/>the benefit is mostly a trained extra token"}}:::finding

    subgraph S8C["Step 8c: teach to imagine with the image as teacher"]
        direction TB
        TF["Training: with prob 0.5 feed the TRUE image embedding<br/>as the imagination token (teacher forcing)"]:::done
        TS["Test: blank slate, own z only (no image anywhere)"]:::done
        TF --> TS
    end
    R8C["final: own 1.0844, shuffled 1.0971, oracle 1.0228,<br/>oracle_wrong 1.0381, constant 1.1038<br/>gain +0.0126 but unstable over epochs"]:::done

    subgraph S8D["Step 8d: 150 fresh questions, paired bootstrap CIs"]
        direction TB
        CoT["CoT setting: image content usable +0.0010 [-0.0004, +0.0025]"]:::done
        AF["Answer-first: own beats blank by 0.0903 but oracle = wrong = own<br/>so any image-like token helps equally"]:::done
    end
    F8D{{"Finding: the text-only LLM did not read image content<br/>the test bed could not show a visual benefit"}}:::finding

    D8 --> S8 --> R8 --> S8B --> F8B --> S8C --> R8C --> S8D --> F8D
```

---

## 5. Advanced track pipeline (A0, A, B, C)

```mermaid
flowchart LR
    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef ckpt fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef planned fill:#eeeeee,stroke:#757575,stroke-dasharray: 5 5,color:#000

    subgraph DATA["Data"]
        direction TB
        COCO20["COCO 20,000 images + captions"]:::data
        V2["Spatial v2 JSON + images<br/>train 40,000 / val 3,000 / test 3,000 / test_ood 2,000<br/>(Phase A and B use 20,000 / 2,000 / 2,000)"]:::data
        FEAT["CLIP features, cached<br/>penultimate patch tokens pooled to 3x3<br/>9 tokens x 768-d per image"]:::data
    end

    A0["A0 alignment<br/>imagine_step8_advanced_a0.py<br/>LLM frozen, train Reader on captions<br/>2 epochs"]:::done
    RA0[("adv_a0_*/reader.pt<br/>one per grid size")]:::ckpt

    PA["Phase A<br/>imagine_step8_advanced_a.py<br/>train Reader + LoRA (r=16) on the answer letter<br/>blank tokens used 15% of the time"]:::done
    CA[("adv_a_*_spatial2/<br/>reader.pt + lora/")]:::ckpt
    EVA["a_eval<br/>imagine_step8_advanced_a_eval.py<br/>no training, per-family accuracy on any split"]:::done

    PB["Phase B<br/>imagine_step8_advanced_b.py --track imagine<br/>train only the Imagination Head<br/>Reader, LoRA, LLM frozen"]:::done
    CB[("adv_b_*_spatial2_imagine/head.pt")]:::ckpt

    PC["Phase C<br/>imagine_step8_advanced_c.py<br/>held-out test and test_ood,<br/>per-group tables, retrieval gallery PNG"]:::done
    OUT["imagine_out/*.png<br/>+ tables with bootstrap CIs"]:::done

    G7["A0 and Phase A for grid 7<br/>(49 tokens x 768-d)"]:::planned

    COCO20 --> A0
    FEAT --> A0
    A0 --> RA0 --> PA
    V2 --> FEAT
    FEAT --> PA
    PA --> CA
    CA --> EVA
    CA --> PB
    FEAT --> PB
    PB --> CB
    CA --> PC
    CB --> PC
    FEAT --> PC
    PC --> OUT
    G7 -.-> PA
```

---

## 6. Core architecture (answer-first, what runs at test time and in training)

```mermaid
flowchart TD
    classDef txt fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef img fill:#fff3c4,stroke:#f9a825,color:#000
    classDef net fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef frozen fill:#e0e0e0,stroke:#616161,color:#000
    classDef outn fill:#ffe0b2,stroke:#e65100,color:#000

    Q["Question + choices (text only)"]:::txt --> TK["Tokenizer<br/>prompt token ids"]:::txt
    TK --> PE["Prompt embeddings"]:::txt
    PE --> LLM1["Qwen2.5-0.5B + LoRA<br/>pass 1: prompt only"]:::frozen
    LLM1 --> H["h = hidden state at the last question token<br/>896-d"]:::txt

    subgraph STUDENT["Imagination path (the blank-slate student: no image)"]
        direction TB
        HEAD["Imagination Head (trained in Phase B)<br/>LayerNorm, MLP 896 to 2048 to 2048 to 6912<br/>20.2 M params"]:::net
        Z["z = 3x3 grid of 768-d imagined CLIP-patch features<br/>9 tokens"]:::img
        HEAD --> Z
    end
    H --> HEAD

    subgraph TEACHER["Teacher path (training only, and the 'oracle' control)"]
        direction TB
        IMG["True image of the question"]:::img
        CLP["CLIP ViT-B/32 (frozen)<br/>penultimate patch tokens, pooled to 3x3"]:::frozen
        TFT["True features 9 x 768"]:::img
        IMG --> CLP --> TFT
    end

    RD["Reader (trained in A0 and Phase A, frozen in Phase B)<br/>LayerNorm, MLP, position embedding, rescale to token size<br/>+ learned blank (null) tokens"]:::frozen
    Z -->|"own: imagined features"| RD
    TFT -.->|"oracle or teacher: true features"| RD
    RD --> VT["9 visual tokens, 896-d each"]:::img

    ANS["Answer prefix tokens: Answer:"]:::txt
    PE --> CAT
    VT --> CAT["Sequence = prompt, visual tokens, Answer:"]:::txt
    ANS --> CAT
    CAT --> LLM2["Qwen2.5-0.5B + LoRA<br/>pass 2: final hidden state at the last token"]:::frozen
    LLM2 --> LG["Logits for letters A to H only<br/>invalid choices masked"]:::net
    LG --> AL["Predicted answer letter<br/>metrics: accuracy and restricted cross-entropy"]:::outn

    NOTE["Training sets: Phase A trains Reader + LoRA,<br/>Phase B trains only the Imagination Head.<br/>Test time: no image, only the student path."]:::txt
    AL -.-> NOTE
```

---

## 7. Phase B: how the Imagination Head is trained

```mermaid
flowchart TD
    classDef trainable fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef frozen fill:#e0e0e0,stroke:#616161,color:#000
    classDef loss fill:#fff3c4,stroke:#f9a825,color:#000
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#000

    HQ["Cached question hidden state h<br/>14,017 imagine-track train items, 896-d"]:::data
    HEAD["Imagination Head (TRAINABLE)<br/>AdamW lr 3e-4, cosine schedule"]:::trainable
    IMAG["imag = predicted 9 x 768 features"]:::data
    HQ --> HEAD --> IMAG

    TIMG["True image"]:::data --> TCLIP["CLIP 3x3 features<br/>layer-normalised target"]:::frozen

    RD["Reader (frozen)"]:::frozen
    LLM["Qwen2.5-0.5B + LoRA (frozen)<br/>gradients flow through it to the head"]:::frozen
    IMAG --> RD --> LLM --> LOWN["logits_own"]:::data
    TCLIP --> RD2["Reader (frozen)"]:::frozen --> LLM2["same LLM, no gradient"]:::frozen --> LORA["logits_oracle (teacher)"]:::data

    IMAG --> LF["feat loss weight 1.0<br/>1 - cosine(imag, target) + 0.1 x MSE"]:::loss
    TCLIP --> LF
    IMAG --> LN["nce loss weight 0.5<br/>symmetric InfoNCE across the batch<br/>pooled and centred, temperature 0.07<br/>(imagination must stay specific to its question)"]:::loss
    TCLIP --> LN
    LOWN --> LA["ans loss weight 1.0<br/>cross-entropy to the gold letter"]:::loss
    LOWN --> LK["kd loss weight 1.0<br/>KL( teacher answer distribution || own answer distribution )"]:::loss
    LORA --> LK

    LF --> SUM["total = feat + 0.5 nce + ans + kd"]:::loss
    LN --> SUM
    LA --> SUM
    LK --> SUM
    SUM -->|"backward: update only the head"| HEAD

    RES["Validation (n=1386): own 94.6 to 95.2%, own_shuffled 95.0 to 92.8%<br/>retrieval top1 0.000 to 0.069, top5 0.001 to 0.235"]:::data
    SUM -.-> RES
```

---

## 8. Evaluation conditions: what each condition feeds into the LLM

```mermaid
flowchart LR
    classDef src fill:#fff3c4,stroke:#f9a825,color:#000
    classDef net fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef res fill:#ffe0b2,stroke:#e65100,color:#000

    subgraph SOURCES["Where the visual tokens come from"]
        direction TB
        O["oracle<br/>TRUE image of this question<br/>CLIP, Reader"]:::src
        W["wrong<br/>image of ANOTHER question<br/>CLIP, Reader"]:::src
        K["const (blank)<br/>learned null tokens"]:::src
        OWN["own<br/>this question: h, Imagination Head, Reader<br/>NO image"]:::src
        OS["own_shuffled<br/>ANOTHER question: h, Imagination Head, Reader"]:::src
    end

    O --> VT["9 visual tokens"]:::net
    W --> VT
    K --> VT
    OWN --> VT
    OS --> VT
    VT --> LLM["prompt, visual tokens, Answer: through LLM"]:::net
    LLM --> M["accuracy and cross-entropy per question"]:::net

    M --> C1["own vs const: does imagination beat blank?"]:::res
    M --> C2["own_shuffled vs own: is the imagination example-specific?"]:::res
    M --> C3["wrong vs own: does it beat a random real image?"]:::res
    M --> C4["wrong vs oracle: can the model use image content at all?"]:::res
    M --> C5["own vs oracle: headroom left"]:::res
    C1 --> STAT["paired bootstrap 95% CI<br/>SUPPORTED if CI above 0, OPPOSITE if below 0,<br/>otherwise not distinguishable from 0"]:::res
    C2 --> STAT
    C3 --> STAT
    C4 --> STAT
    C5 --> STAT

    TEST["Phase C results (accuracy %)<br/>test: oracle 97.3, own 96.7, const 96.5, own_shuffled 94.7, wrong 91.7<br/>test_ood: oracle 93.7, own 85.9, const 85.3, own_shuffled 85.1, wrong 82.8<br/>triangle on test_ood: oracle 98.2, own 51.7, const 52.0"]:::res
    STAT -.-> TEST
```

---

## 9. Datasets and the leak audit

```mermaid
flowchart TD
    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef bad fill:#ffcdd2,stroke:#c62828,color:#000
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef finding fill:#ffe0b2,stroke:#e65100,color:#000

    subgraph SQ["ScienceQA (image questions)"]
        direction TB
        SQD["derek-thomas/ScienceQA<br/>train / validation / test"]:::data
        SQX["imagine_diagnose_data.py<br/>63% have a hint, 48% long hints,<br/>12% to 14% contain the gold string"]:::done
        SQV["Verdict: text-solvable, oracle = wrong"]:::bad
        SQD --> SQX --> SQV
    end

    subgraph V1["Spatial v1 (built by the Cursor agent, imagine_build_spatial_cot.py)"]
        direction TB
        V1G["80,000 / 4,000 / 4,000 items<br/>8 families: fit_ball_box, fit_rect, triangle, compare_len,<br/>clock, spatial_imagine, count_see, mark_see"]:::data
        V1A["imagine_audit_data.py: 8 of 8 flagged<br/>image shows fits / does not fit text and green or red colour<br/>ring on the gold object<br/>mark_see answer always index 0<br/>count_see choices = gold-1 .. gold+2"]:::bad
        V1P["Phase A: oracle 100.0, wrong 58.3, blank 99.2<br/>text alone solves it, the pass is meaningless"]:::bad
        V1G --> V1A --> V1P
    end

    subgraph V2["Spatial v2 (imagine_build_spatial_v2.py), leak-free design rules"]
        direction TB
        R1["Rules: picture shows scene or state, never the verdict<br/>yes/no balanced by construction<br/>choices shuffled, answer index uniform<br/>distractors independent of the gold"]:::done
        subgraph IMGT["imagine track: the text fully specifies the scene"]
            direction LR
            F1["fit_ball_box"]:::data
            F2["fit_rect"]:::data
            F3["triangle"]:::data
            F4["spatial_grid"]:::data
            F5["grid_walk<br/>multi-step state tracking"]:::data
        end
        subgraph SEET["see track: the image is necessary"]
            direction LR
            F6["count_see"]:::data
            F7["mark_see"]:::data
            F8["rel_see"]:::data
        end
        SP["Splits: train 40,000 / val 3,000 / test 3,000 / test_ood 2,000<br/>test_ood = bigger numbers, bigger grids, longer walks (imagine families only)"]:::data
        AUD["Self-audit and imagine_audit_data.py: 0 of 8 flagged<br/>notes: to-scale sketches still encode geometry"]:::done
        R1 --> IMGT
        R1 --> SEET
        IMGT --> SP
        SEET --> SP
        SP --> AUD
    end

    SQV -->|"need a task where the image is necessary"| V1
    V1P -->|"audit exposes the leaks"| V2
    AUD --> USE["CLIP 3x3 features cached, used by Phase A, B, C"]:::done
```

---

## 10. Planned: multi-step `grid_walk` chain (not yet implemented)

```mermaid
flowchart TD
    classDef planned fill:#eeeeee,stroke:#757575,stroke-dasharray: 5 5,color:#000
    classDef known fill:#d4f4dd,stroke:#2e7d32,color:#000

    DATA["Rebuild v2 with step images<br/>imagine_build_spatial_v2.py --step_images<br/>one board image per move"]:::planned
    Q["Question: grid, walls, start, move list m1 to mT"]:::planned
    Q --> H0["LLM reads the question: h0"]:::planned
    H0 --> Z0["Imagination Head: initial board features z0"]:::planned

    subgraph LOOP["For t = 1 to T: per-step imagination (the PDF loop)"]
        direction TB
        ZP["z t-1: previous imagined board"]:::planned
        MV["move m t"]:::planned
        UPD["Imagination Head update<br/>input: z t-1 and m t"]:::planned
        ZT["z t: imagined board after move t"]:::planned
        ZP --> UPD
        MV --> UPD
        UPD --> ZT
        ZT --> ZP
        TEACH["Training teacher: true board t features<br/>(teacher forcing / scheduled sampling)"]:::planned
        ZT -.-> TEACH
        DIV["Anti-collapse loss<br/>cosine(z t, z t-1) below a margin,<br/>plus measure collapse across steps"]:::planned
        ZT --> DIV
    end
    Z0 --> ZP
    DATA --> TEACH

    GATE["Gate g: trained on the empirical target<br/>1 if CE(const) greater than CE(own), else 0"]:::planned
    ZT --> BLEND["tokens = g x own + (1 - g) x blank"]:::planned
    GATE --> BLEND
    BLEND --> RD["Reader, visual tokens"]:::known
    RD --> LLM["LLM answers the final cell (row, column)"]:::known
    LLM --> EVAL["Evaluate on test and longer test_ood walks<br/>where text-only was about 90% and 76%"]:::planned

    WHY["Why grid_walk: the static starting board did not help<br/>oracle 90.1 vs const 90.1 on test.<br/>One local board update per move may beat text-only memory drift."]:::planned
    EVAL -.-> WHY
```

---

## 11. Future idea: dual-stack with bidirectional cross-attention (not started)

```mermaid
flowchart TD
    classDef planned fill:#eeeeee,stroke:#757575,stroke-dasharray: 5 5,color:#000

    TIN["Text tokens (problem + reasoning so far)"]:::planned
    LIN["Latent image tokens (the imagination so far)"]:::planned

    subgraph BLK1["Block 1"]
        direction TB
        TS1["Text stack: self-attention"]:::planned
        LS1["Image stack: self-attention"]:::planned
        XA1["Cross-attention in BOTH directions<br/>text attends to image, image attends to text"]:::planned
        TS1 --> XA1
        LS1 --> XA1
    end

    subgraph BLK2["Block 2 ... Block N (repeat)"]
        direction TB
        TS2["Text stack: self-attention"]:::planned
        LS2["Image stack: self-attention"]:::planned
        XA2["Bidirectional cross-attention"]:::planned
        TS2 --> XA2
        LS2 --> XA2
    end

    TIN --> TS1
    LIN --> LS1
    XA1 --> TS2
    XA1 --> LS2
    XA2 --> TOUT["Text output: next token / answer"]:::planned
    XA2 --> LOUT["Updated latent image tokens (next imagination)"]:::planned
    LOUT -.->|"fed back at the next reasoning step"| LIN

    WHY["Motivation from Phase B and C:<br/>a single imagination made from the question's final hidden state cannot<br/>compute what that state could not (OOD triangle: 51.7%).<br/>Layer-by-layer exchange lets the imagination use intermediate text computation."]:::planned
    XA2 -.-> WHY
```

---

## 12. Programs and checkpoints map

```mermaid
flowchart LR
    classDef mod fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef run fill:#d4f4dd,stroke:#2e7d32,color:#000
    classDef ckpt fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef never fill:#eeeeee,stroke:#757575,stroke-dasharray: 5 5,color:#000
    classDef tool fill:#fff3c4,stroke:#f9a825,color:#000

    subgraph SHARED["Shared modules (imported, not run)"]
        direction TB
        DATA["imagine_data.py<br/>COCO + ScienceQA download, CLIP"]:::mod
        LIB["imagine_adv_lib.py<br/>loaders, VisSet, Reader, Imagination Head,<br/>answer-first forward, eval and bootstrap"]:::mod
    end

    subgraph T1["Track 1 scripts (SmolLM2-135M)"]
        direction TB
        S1["step1: backbone"]:::run
        S2["step2: head + gate"]:::run
        S3["step3: anti-collapse"]:::run
        S4["step4: FeedbackChain"]:::run
        S5["step5: COCO alignment"]:::run
        S6["step6: gate"]:::run
        S7["step7: retrieval"]:::run
        S8["step8: joint training"]:::run
        S8B["step8b: controls"]:::run
        S8C["step8c: teacher forcing"]:::run
        S8D["step8d: significance"]:::run
        S9["step9: LoRA joint"]:::never
        S10["step10: GRPO"]:::never
    end

    subgraph T2["Advanced track scripts (Qwen2.5-0.5B)"]
        direction TB
        A0["step8_advanced_a0"]:::run
        A["step8_advanced_a"]:::run
        AE["step8_advanced_a_eval"]:::run
        B["step8_advanced_b"]:::run
        C["step8_advanced_c"]:::run
    end

    subgraph TOOLS["Dataset and tooling scripts"]
        direction TB
        DG["diagnose_data (Cursor)"]:::tool
        BV1["build_spatial_cot (Cursor, v1 leaky)"]:::tool
        FP["fetch_public (Cursor, not used)"]:::never
        AU["audit_data"]:::tool
        BV2["build_spatial_v2"]:::tool
        CE["check_env"]:::tool
    end

    subgraph CK["Checkpoints and caches"]
        direction TB
        H5[("head_step5.pt")]:::ckpt
        H6[("head_step6.pt")]:::ckpt
        CH8[("chain_step8.pt")]:::ckpt
        CH8C[("chain_step8c.pt + lora_step8c")]:::ckpt
        RA0[("adv_a0_*/reader.pt")]:::ckpt
        CA[("adv_a_*/reader.pt + lora")]:::ckpt
        CB[("adv_b_*/head.pt")]:::ckpt
        SJ[("spatial_cot_v2/*.json + images")]:::ckpt
        FE[("adv_feats_* (CLIP 3x3 features)")]:::ckpt
    end

    DATA --> S1
    DATA --> S5
    DATA --> S6
    DATA --> S8
    DATA --> A0
    LIB --> A0
    LIB --> A
    LIB --> AE
    LIB --> B
    LIB --> C

    S1 --> S2 --> S3 --> S4 --> S5
    S5 --> H5 --> S6 --> H6
    H6 --> S7
    H6 --> S8
    S8 --> CH8 --> S8B
    S8 --> S8C
    S8C --> CH8C --> S8D
    CH8 -.-> S9
    S9 -.-> S10

    A0 --> RA0 --> A
    BV2 --> SJ
    BV1 -.->|"superseded"| BV2
    SJ --> AU
    SJ --> FE
    FE --> A0
    FE --> A
    A --> CA
    CA --> AE
    CA --> B
    B --> CB
    CA --> C
    CB --> C
    DG -.-> BV1
    CE -.->|"checks versions of"| LIB
```
