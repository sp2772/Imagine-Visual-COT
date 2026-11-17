# Step 10: GRPO on the final answer (accuracy + format rewards), with the anti-collapse term kept as an auxiliary loss.
# Rollout = [prompt] -> n_cot sampled "Step k:" segments (imagination fed back between them) -> "Answer:" + sampled letter.
# EXPECT: a 135M model stays near chance accuracy; the visible gain is the FORMAT reward (it learns to emit a valid letter).
# Run: python imagine_step10.py --iters 30 --prompts 2 --G 6      (needs step 9)
import argparse, os, random, re, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, load_chain_parts, save_chain_parts
from imagine_step8 import build_prompt, LETTERS, amp, cons_cos
from imagine_data import *

def sample_token(logits, temp):
    if temp <= 0: return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits.float() / temp, -1), 1))

@torch.no_grad()
def sample_rollout(chain, tok, item, step_end, n_cot, max_tok, temp, use_amp):
    dev = chain.img_start.device; model = chain.model; embed = model.get_input_embeddings(); st = {"past": None}
    def feed(emb):
        out = model(inputs_embeds=emb, past_key_values=st["past"], use_cache=True, output_hidden_states=True)
        st["past"] = out.past_key_values; return out
    emb_of = lambda ids: embed(torch.tensor([ids], device=dev))
    def imagine(out):
        z, g = chain.head(out.hidden_states[-1][0, -1].float()); return chain.proj(z).view(1, 1, -1), z, g
    nl = tok("\n", add_special_tokens=False).input_ids[-1]
    with amp(use_amp):
        prompt = tok(build_prompt(item), add_special_tokens=False).input_ids
        out = feed(torch.cat([chain.img_start.view(1, 1, -1), emb_of(prompt)], 1))
        prev_i, z, g = imagine(out); segs, masks, gates = [prompt], [[False] * len(prompt)], [g.item()]
        for s in range(n_cot):
            out = feed(prev_i.to(out.hidden_states[-1].dtype))
            pre = tok(f"Step {s+1}:", add_special_tokens=False).input_ids
            out = feed(emb_of(pre)); ids, msk = list(pre), [False] * len(pre)
            for _ in range(max_tok):
                t = sample_token(out.logits[0, -1], temp); ids.append(t); msk.append(True); out = feed(emb_of([t]))
                if t in step_end: break
            else:
                ids.append(nl); msk.append(False); out = feed(emb_of([nl]))
            prev_i, z, g = imagine(out); segs.append(ids); masks.append(msk); gates.append(g.item())
        out = feed(prev_i.to(out.hidden_states[-1].dtype))
        pre = tok("Answer:", add_special_tokens=False).input_ids; out = feed(emb_of(pre)); ans = []
        for _ in range(3):
            t = sample_token(out.logits[0, -1], temp); ans.append(t); out = feed(emb_of([t]))
            if t in step_end: break
        segs.append(pre + ans); masks.append([False] * len(pre) + [True] * len(ans))
    m = re.search(f"[A-{LETTERS[len(item['choices'])-1]}]", tok.decode(ans))
    return {"segs": [torch.tensor(s) for s in segs], "masks": [torch.tensor(m_, dtype=torch.bool) for m_ in masks],
            "letter": m.group(0) if m else None, "gates": gates, "text": [tok.decode(s) for s in segs[1:]]}

def reward(ro, item):
    fmt = 0.2 if ro["letter"] is not None else 0.0
    acc = 1.0 if (ro["letter"] is not None and LETTERS.index(ro["letter"]) == item["answer"]) else 0.0
    return acc + fmt, acc, fmt

def eval_greedy(chain, tok, items, step_end, n_cot, max_tok, use_amp, desc):
    acc = fmt = 0.0
    for it in tqdm(items, desc=desc, leave=False):
        ro = sample_rollout(chain, tok, it, step_end, n_cot, max_tok, 0.0, use_amp); _, a_, f_ = reward(ro, it); acc += a_; fmt += f_ / 0.2
    return acc / len(items), fmt / len(items)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--prompts", type=int, default=2); ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--n_cot", type=int, default=2); ap.add_argument("--max_tok", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n_eval", type=int, default=40); ap.add_argument("--div_w", type=float, default=0.1)
    a = ap.parse_args(); set_seed(0)
    try: from peft import PeftModel
    except ImportError: raise SystemExit("pip install peft")
    banner("STEP 10: GRPO")
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    tok, model = load_backbone(); d = model.config.hidden_size
    chain = FeedbackChain(model, d).to(device)
    if os.path.isdir(f"{CKPT}/lora_step9"):
        chain.model = PeftModel.from_pretrained(chain.model, f"{CKPT}/lora_step9", is_trainable=True).to(device); load_chain_parts(chain, f"{CKPT}/chain_step9.pt"); print("loaded step 9 LoRA + head")
    else:
        print("WARNING: step 9 missing; GRPO on the plain frozen model would have nothing to train. Run step 9 first."); raise SystemExit
    chain.eval()
    items = [it for it in get_scienceqa(n_img=0, n_noimg=a.iters * a.prompts + a.n_eval + 50) if not it["has_image"]]
    ev_items, tr_items = items[:a.n_eval], items[a.n_eval:]
    chance = sum(1 / len(it["choices"]) for it in ev_items) / len(ev_items)
    step_end = {i for i in tqdm(range(len(tok)), desc="finding newline tokens", leave=False) if "\n" in tok.decode([i])}
    print(f"train questions {len(tr_items)} | eval questions {len(ev_items)} | chance accuracy ~ {chance:.3f}")
    acc0, fmt0 = eval_greedy(chain, tok, ev_items, step_end, a.n_cot, a.max_tok, use_amp, "eval before")
    print(f"BEFORE GRPO: greedy accuracy {acc0:.3f} | valid-answer-format rate {fmt0:.3f}")
    ro = sample_rollout(chain, tok, ev_items[0], step_end, a.n_cot, a.max_tok, a.temp, use_amp)
    print("sample rollout:", [t.strip() for t in ro["text"]], "| gates:", [round(x, 2) for x in ro["gates"]])

    params = [p for p in chain.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    pbar = tqdm(range(a.iters), desc="GRPO")
    for it_i in pbar:
        batch = random.sample(tr_items, a.prompts); work, R, A, F_ = [], [], [], []
        for item in batch:
            group = [sample_rollout(chain, tok, item, step_end, a.n_cot, a.max_tok, a.temp, use_amp) for _ in range(a.G)]
            rs = torch.tensor([reward(g, item)[0] for g in group]); adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            R += rs.tolist(); A += [reward(g, item)[1] for g in group]; F_ += [reward(g, item)[2] for g in group]
            work += [(g, float(x)) for g, x in zip(group, adv) if abs(float(x)) > 1e-3]
        if work:
            opt.zero_grad(); cos_log = []
            for g, adv in work:
                with amp(use_amp): r = chain(g["segs"], g["masks"])
                logp = torch.cat([lp[m.to(lp.device)] for lp, m in zip(r["lps"], g["masks"])]).mean()
                div = F.relu(cons_cos(r["z"]) - 0.8).mean(); cos_log.append(cons_cos(r["z"]).mean().item())
                ((-adv * logp + a.div_w * div) / len(work)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        pbar.set_postfix(reward=f"{sum(R)/len(R):.3f}", acc=f"{sum(A)/len(A):.3f}", fmt=f"{sum(F_)/len(F_)/0.2:.2f}", active=f"{len(work)}/{len(R)}")
    acc1, fmt1 = eval_greedy(chain, tok, ev_items, step_end, a.n_cot, a.max_tok, use_amp, "eval after")
    print(f"\nAFTER  GRPO: greedy accuracy {acc1:.3f} | valid-answer-format rate {fmt1:.3f}   (chance ~ {chance:.3f})")
    ro = sample_rollout(chain, tok, ev_items[0], step_end, a.n_cot, a.max_tok, a.temp, use_amp)
    print("sample rollout:", [t.strip() for t in ro["text"]], "| gates:", [round(x, 2) for x in ro["gates"]])
    chain.model.save_pretrained(f"{CKPT}/lora_step10"); save_chain_parts(chain, f"{CKPT}/chain_step10.pt")
    print(f"saved {CKPT}/lora_step10\nSTEP 10 OK")

if __name__ == "__main__":
    main()
