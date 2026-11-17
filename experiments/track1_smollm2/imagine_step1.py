# Step 1: load a small LLM backbone and extract hidden states h_t.
# Install once:  pip install torch transformers datasets pillow tqdm matplotlib peft requests
# Run:           python imagine_step1.py
import os, time, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("IMAGINE_MODEL", "HuggingFaceTB/SmolLM2-135M")   # d_model = 576, CPU friendly
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_backbone():
    print(f"[load] backbone = {MODEL} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[load] d_model={model.config.hidden_size} layers={model.config.num_hidden_layers} params={n:.1f}M", flush=True)
    return tok, model

@torch.no_grad()
def get_hidden(tok, model, text):
    """Last-layer hidden states [1, seq_len, d_model]."""
    inp = tok(text, return_tensors="pt").to(device)
    return model(**inp, output_hidden_states=True).hidden_states[-1]

@torch.no_grad()
def batch_hidden(tok, model, texts, bs=64, pool="last", max_len=128, desc="hidden states"):
    """One vector per text: hidden state at the last real token (pool='last') or mean (pool='mean'). -> [N, d] on CPU."""
    tok.padding_side = "right"; outs = []
    for i in tqdm(range(0, len(texts), bs), desc=desc):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        hid = model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc.attention_mask
        if pool == "last": v = hid[torch.arange(hid.size(0), device=device), m.sum(1) - 1]
        else: v = (hid * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        outs.append(v.float().cpu())
    return torch.cat(outs)

if __name__ == "__main__":
    t0 = time.time()
    tok, model = load_backbone()
    print("device:", device, "| load time: %.1fs" % (time.time() - t0))

    prompt = "Can a ball of diameter 9 cm fit inside a box 10 cm wide and 8 cm tall?\nStep 1:"
    h = get_hidden(tok, model, prompt)
    print("hidden states shape:", tuple(h.shape), " (expect [1, seq_len, d_model])")
    h_t = h[:, -1]
    print("h_t shape:", tuple(h_t.shape), "| norm: %.2f" % h_t.norm().item())

    hb = batch_hidden(tok, model, [prompt, "What is the capital of France?\n"], bs=2)
    print("batch_hidden shape:", tuple(hb.shape), "(expect (2, d_model))")

    inp = tok(prompt, return_tensors="pt").to(device)
    gen = model.generate(**inp, max_new_tokens=20, do_sample=False)
    print("generation:", repr(tok.decode(gen[0], skip_special_tokens=True)))
    print("\nSTEP 1 OK" )
