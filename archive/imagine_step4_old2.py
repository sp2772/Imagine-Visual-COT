# Step 4: the feedback loop (the "causal" part).
# Context at reasoning step t:  [I_-1, seg_0, I_0, seg_1, ..., I_{t-1}, seg_t]
#   I_{t-1} = proj(z_{t-1})  is the previous imagination injected as ONE extra input embedding.
#   h_t = hidden state at the last token of seg_t  ->  z_t, g_t = head(h_t)
import torch, torch.nn as nn, torch.nn.functional as F
from imagine_step1 import load_backbone, device
from imagine_step2 import ImaginationHead, D_VIS
from imagine_data import banner, set_seed, CKPT

class FeedbackChain(nn.Module):
    def __init__(self, model, d_model, d_vis=D_VIS):
        super().__init__()
        self.model = model
        self.head = ImaginationHead(d_model, d_vis)
        self.proj = nn.Linear(d_vis, d_model)               # z (visual space) -> LLM embedding space
        self.img_start = nn.Parameter(torch.randn(d_model) * 0.02)   # I_-1, used before any imagination exists

    def forward(self, segs, masks=None, mode="feedback", z_swap=None):
        """segs: list of LongTensor token ids (one per reasoning step).
        masks: optional list of bool tensors; True = token counts for the LM loss.
        mode 'feedback': I_t = proj(z_t).  mode 'constant': I_t = img_start (ablation: imagination tokens carry no info)."""
        dev = self.img_start.device
        embed = self.model.get_input_embeddings()
        ctx, prev_i = [], self.img_start.view(1, 1, -1)
        hs, zs, gs, lps = [], [], [], []
        for ids in segs:
            ids = ids.to(dev).view(1, -1)
            e = embed(ids)
            ctx += [prev_i.to(e.dtype), e]
            inp = torch.cat(ctx, 1)
            out = self.model(inputs_embeds=inp, output_hidden_states=True)
            h_t = out.hidden_states[-1][0, -1].float()
            z, g = self.head(h_t)
            n, L = ids.size(1), inp.size(1)
            logp = F.log_softmax(out.logits[0, L - n - 1:L - 1].float(), -1)   # positions that predict seg tokens
            lps.append(logp.gather(1, ids[0].unsqueeze(1)).squeeze(1))
            hs.append(h_t); zs.append(z); gs.append(g)
            zz = z if z_swap is None else z_swap[min(len(zs) - 1, len(z_swap) - 1)].to(z.device)
            prev_i = self.proj(zz).view(1, 1, -1) if mode == "feedback" else self.img_start.view(1, 1, -1)
        tot, cnt = 0.0, 0
        for i, lp in enumerate(lps):
            m = torch.ones_like(lp, dtype=torch.bool) if masks is None else masks[i].to(lp.device)
            tot = tot - lp[m].sum(); cnt += int(m.sum())
        return {"h": torch.stack(hs), "z": torch.stack(zs), "g": torch.stack(gs), "lps": lps,
                "lm_loss": tot / max(cnt, 1)}

def make_segs(tok, texts):
    segs = []
    for t in texts:
        if not t.endswith("\n"): t += "\n"
        segs.append(torch.tensor(tok(t, add_special_tokens=False).input_ids, dtype=torch.long))
    return segs

def save_chain_parts(chain, path):
    torch.save({"head": chain.head.state_dict(), "proj": chain.proj.state_dict(), "img_start": chain.img_start.data.cpu()}, path)

def load_chain_parts(chain, path):
    ck = torch.load(path, map_location="cpu")
    chain.head.load_state_dict(ck["head"]); chain.proj.load_state_dict(ck["proj"])
    chain.img_start.data.copy_(ck["img_start"])

if __name__ == "__main__":
    set_seed(0)
    banner("STEP 4: feedback loop")
    tok, model = load_backbone()
    d = model.config.hidden_size
    chain = FeedbackChain(model, d).to(device)
    texts = ["Question: Can a ball of 9 cm diameter fit in a box 10 cm wide and 8 cm tall?",
             "Step 1: The box is 10 cm wide and 8 cm tall.",
             "Step 2: The ball needs 9 cm of room in every direction.",
             "Step 3: 9 cm is less than 10 cm, so it fits in width.",
             "Step 4: 9 cm is more than 8 cm, so it does not fit in height.",
             "Answer: No, the ball cannot fit."]
    segs = make_segs(tok, texts)
    print("tokens per step:", [len(s) for s in segs])
    emb_norm = model.get_input_embeddings().weight.norm(dim=1).mean().item()
    with torch.no_grad():
        r = chain(segs, mode="feedback")
        pn = chain.proj(r["z"]).norm(dim=-1).mean().item()
    print(f"avg token-embedding norm: {emb_norm:.3f} | avg injected imagination-token norm: {pn:.3f} (same order of magnitude is good)")

    print("\n-- per-step view (random-init head, so values are meaningless but shapes/flow must work) --")
    z, g = r["z"], r["g"]
    for t in range(len(segs)):
        c = F.cosine_similarity(z[t], z[t - 1], dim=0).item() if t else float("nan")
        print(f"step {t}: gate={g[t].item():.3f}  cos(z_t, z_t-1)={c:.3f}  h_t norm={r['h'][t].norm().item():.1f}")
    print("shapes: h", tuple(r["h"].shape), "z", tuple(r["z"].shape), "g", tuple(r["g"].shape))

    print("\n-- check 1: imagination feedback really changes the LLM (feedback vs constant token) --")
    with torch.no_grad():
        rc = chain(segs, mode="constant")
    rel = (r["h"][-1] - rc["h"][-1]).norm() / rc["h"][-1].norm()
    print(f"LM loss feedback={r['lm_loss'].item():.4f}  constant={rc['lm_loss'].item():.4f}  | rel. change of final h: {rel.item():.4f} (must be > 0)")

    print("\n-- check 2: causality (editing the LAST step must not change earlier z) --")
    segs2 = list(segs); segs2[-1] = make_segs(tok, ["Answer: Yes, it fits easily."])[0]
    with torch.no_grad():
        r2 = chain(segs2)
    print("max |dz| on earlier steps:", f"{(r['z'][:-1] - r2['z'][:-1]).abs().max().item():.2e}", "(expect ~0)")
    print("max |dz| on last step    :", f"{(r['z'][-1] - r2['z'][-1]).abs().max().item():.2e}", "(expect > 0)")

    print("\n-- check 3: gradients flow back through the loop --")
    chain.zero_grad(); out = chain(segs); out["lm_loss"].backward()
    gn = lambda p: 0.0 if p.grad is None else p.grad.norm().item()
    print(f"grad norm W2 (head)   : {gn(chain.head.W2.weight):.3e}  (expect > 0: LM loss reaches the head via the loop)")
    print(f"grad norm proj        : {gn(chain.proj.weight):.3e}  (expect > 0)")
    print(f"grad norm W1 (gate)   : {gn(chain.head.W1.weight):.3e}  (expect 0: gate is trained separately, step 6)")
    print("\nSTEP 4 OK")
