# Step 8c: learn to IMAGINE ON ITS OWN. Images are only a TEACHING signal (like teacher forcing in next-token training):
#   training : with prob p_tf the true image embedding is fed in as the imagination token (teacher forcing),
#              otherwise the model's own z is fed back (scheduled sampling). Plus alignment losses pull z toward the image.
#   test     : blank slate. The model sees only the question text and feeds back its OWN z. No image anywhere.
# Evaluation on image questions compares (LM loss, lower = better):
#   own (test-time condition) | own_shuffled (another example's z) | oracle (true image) | oracle_wrong (another image) | constant
#   visual_info_usable = oracle_wrong - oracle      -> can the LLM use image content at all through this interface?
#   own_info_gain      = own_shuffled - own         -> does the model's self-made imagination help THIS example?
#   recovery           = own_info_gain / visual_info_usable
# Run: python imagine_step8c.py                  (LoRA on by default if peft is installed; --no_lora for frozen LLM)
import argparse, os, random, torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, save_chain_parts
from imagine_step8 import load_data, cons_cos, W
from imagine_data import *

def batch_losses(chain, batch, p_tf):
    Z, T, lm, div, gate, imgl, n_tf = [], [], [], [], [], [], 0
    for d in batch:
        tf = d["it"]["has_image"] and random.random() < p_tf
        r = chain(d["segs"], d["masks"], oracle=d["img_emb"] if tf else None); n_tf += int(tf)
        z, n = r["z"], d["n_steps"]
        Z.append(z[1:1 + n]); T.append(d["step_emb"].to(z.device)); lm.append(r["lm_loss"])
        div.append(F.relu(cons_cos(z) - W["margin"]).mean())
        gate.append(F.binary_cross_entropy(r["g"].clamp(1e-5, 1 - 1e-5), torch.full_like(r["g"], float(d["it"]["has_image"]))))
        if d["it"]["has_image"]:
            im = d["img_emb"].to(z.device)
            imgl.append(0.5 * (1 - (F.normalize(z[:1 + n].mean(0), dim=-1) * im).sum()) + 0.5 * (1 - (z[0] * im).sum()))  # z[0] = blank-slate imagination
    Z, T = torch.cat(Z), torch.cat(T)
    logits = (Z @ T.T) / 0.07; tgt = torch.arange(len(Z), device=Z.device)
    align = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
    parts = {"lm": torch.stack(lm).mean(), "align": align, "div": torch.stack(div).mean(), "gate": torch.stack(gate).mean(),
             "img": torch.stack(imgl).mean() if imgl else torch.zeros((), device=Z.device)}
    total = parts["lm"] + W["align"] * align + 1.0 * parts["img"] + W["div"] * parts["div"] + W["gate"] * parts["gate"]
    out = {k: v.item() for k, v in parts.items()}; out["tf"] = n_tf
    return total, out

@torch.no_grad()
def evaluate(chain, val, tag):
    chain.eval(); img = [d for d in val if d["it"]["has_image"]]; txt = [d for d in val if not d["it"]["has_image"]]; m = len(img)
    own = [chain(d["segs"], d["masks"]) for d in tqdm(img, desc=f"{tag}: own", leave=False)]
    zs = [r["z"] for r in own]; res = {"own": [r["lm_loss"].item() for r in own]}
    res["own_shuffled"] = [chain(d["segs"], d["masks"], z_swap=zs[(i + 1) % m])["lm_loss"].item() for i, d in enumerate(tqdm(img, desc=f"{tag}: shuffled", leave=False))]
    res["oracle"] = [chain(d["segs"], d["masks"], oracle=d["img_emb"])["lm_loss"].item() for d in tqdm(img, desc=f"{tag}: oracle", leave=False)]
    res["oracle_wrong"] = [chain(d["segs"], d["masks"], oracle=img[(i + 1) % m]["img_emb"])["lm_loss"].item() for i, d in enumerate(tqdm(img, desc=f"{tag}: wrong image", leave=False))]
    res["constant"] = [chain(d["segs"], d["masks"], mode="constant")["lm_loss"].item() for d in tqdm(img, desc=f"{tag}: constant", leave=False)]
    mean = {k: sum(v) / m for k, v in res.items()}
    Z0 = torch.stack([z[0] for z in zs]); I = torch.stack([d["img_emb"].to(Z0.device) for d in img])
    top1 = ((Z0 @ I.T).argmax(1) == torch.arange(m, device=Z0.device)).float().mean().item()
    cos_right = (Z0 * I).sum(-1).mean().item(); cos_wrong = (Z0 * I.roll(1, 0)).sum(-1).mean().item()
    usable = mean["oracle_wrong"] - mean["oracle"]; gain = mean["own_shuffled"] - mean["own"]
    txt_gap = None
    if txt:
        a = sum(chain(d["segs"], d["masks"])["lm_loss"].item() for d in txt) / len(txt)
        b = sum(chain(d["segs"], d["masks"], mode="constant")["lm_loss"].item() for d in txt) / len(txt); txt_gap = (a, b)
    print(f"\n[{tag}] LM loss on {m} image questions: " + " | ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    print(f"[{tag}] visual_info_usable (oracle_wrong - oracle) = {usable:+.4f}   <- can the LLM use image content at all?")
    print(f"[{tag}] own_info_gain      (own_shuffled - own)    = {gain:+.4f}   <- does self-made imagination help THIS example?")
    print(f"[{tag}] recovery           (gain / usable)         = {gain / usable:+.2f}" if abs(usable) > 1e-6 else f"[{tag}] recovery n/a")
    print(f"[{tag}] blank-slate z_0 vs true image: retrieval top1={top1:.3f} (chance {1/m:.3f}) | cos right={cos_right:.3f} vs wrong={cos_wrong:.3f}")
    if txt_gap: print(f"[{tag}] text-only questions: own={txt_gap[0]:.4f} constant={txt_gap[1]:.4f}", flush=True)
    return {"usable": usable, "gain": gain, "top1": top1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=480); ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--p_tf", type=float, default=0.5, help="prob. of teacher-forcing the true image on image questions")
    ap.add_argument("--k", type=int, default=4, help="imagination tokens injected per step")
    ap.add_argument("--no_lora", action="store_true"); ap.add_argument("--r", type=int, default=8); ap.add_argument("--lr_lora", type=float, default=2e-4)
    a = ap.parse_args(); set_seed(0)
    banner("8c-a: data"); tok, model = load_backbone(); d_model = model.config.hidden_size
    train, val = load_data(a.n_train, a.n_val, tok)
    print(f"train {len(train)} ({sum(x['it']['has_image'] for x in train)} image) | val {len(val)} ({sum(x['it']['has_image'] for x in val)} image)")
    banner("8c-b: model"); chain = FeedbackChain(model, d_model, k=a.k).to(device)
    ck = f"{CKPT}/head_step6.pt" if os.path.exists(f"{CKPT}/head_step6.pt") else f"{CKPT}/head_step5.pt"
    if os.path.exists(ck): chain.head.load_state_dict(torch.load(ck, map_location=device)["head"]); print("loaded head from", ck)
    lora = []
    if not a.no_lora:
        try:
            from peft import LoraConfig, get_peft_model
            chain.model = get_peft_model(chain.model, LoraConfig(r=a.r, lora_alpha=2 * a.r, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
            chain.to(device); lora = [p for p in chain.model.parameters() if p.requires_grad]; print("LoRA on: trainable LoRA params =", sum(p.numel() for p in lora))
        except ImportError: print("peft not installed -> frozen LLM (pip install peft to enable LoRA)")
    groups = [{"params": list(chain.head.parameters()), "lr": 2e-4}, {"params": list(chain.proj.parameters()) + [chain.img_start], "lr": 1e-3}]
    if lora: groups.append({"params": lora, "lr": a.lr_lora})
    params = [p for g in groups for p in g["params"]]; opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    print(f"imagination tokens per step k={a.k} | teacher-forcing prob p_tf={a.p_tf}")
    banner("8c-c: BEFORE training"); evaluate(chain, val, "before")
    banner("8c-d: training")
    for ep in range(a.epochs):
        random.shuffle(train); chain.train()
        pbar = tqdm(range(0, len(train) - a.bs + 1, a.bs), desc=f"epoch {ep+1}/{a.epochs}")
        for i in pbar:
            opt.zero_grad(); loss, parts = batch_losses(chain, train[i:i + a.bs], a.p_tf)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            pbar.set_postfix({k: (f"{v:.3f}" if k != "tf" else v) for k, v in parts.items()})
        evaluate(chain, val, f"epoch {ep+1}")
    banner("8c-e: final"); res = evaluate(chain, val, "final")
    print("\nHOW TO READ: usable>0 = LLM can read the image signal. gain>0 = the model's OWN blank-slate imagination carries example-specific, helpful info.")
    print("             If usable ~ 0 the interface is too weak (try --k 8, more epochs, LoRA) before judging imagination itself.")
    save_chain_parts(chain, f"{CKPT}/chain_step8c.pt")
    if lora: chain.model.save_pretrained(f"{CKPT}/lora_step8c")
    print(f"saved {CKPT}/chain_step8c.pt\nSTEP 8c OK")

if __name__ == "__main__":
    main()
