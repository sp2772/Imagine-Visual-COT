# Step 9: LoRA fine-tuning of the LLM jointly with head/proj (bf16 autocast on GPU), same losses as step 8.
# Run: python imagine_step9.py --epochs 2          (needs step 8 and `pip install peft`)
import argparse, os, torch
from imagine_step1 import load_backbone, device
from imagine_step4 import FeedbackChain, load_chain_parts, save_chain_parts
from imagine_step8 import load_data, evaluate, train_loop, show
from imagine_data import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--lr_lora", type=float, default=2e-4)
    a = ap.parse_args(); set_seed(0)
    try: from peft import LoraConfig, get_peft_model
    except ImportError: raise SystemExit("pip install peft")
    banner("STEP 9: LoRA + joint training")
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported(); print("bf16 autocast:", use_amp)
    tok, model = load_backbone(); d = model.config.hidden_size
    train, val = load_data(a.n_train, a.n_val, tok)
    chain = FeedbackChain(model, d).to(device)
    p8 = f"{CKPT}/chain_step8.pt"
    if os.path.exists(p8): load_chain_parts(chain, p8); print("loaded", p8)
    else: print("WARNING: step 8 checkpoint missing, starting from untrained head")
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
    chain.model = get_peft_model(chain.model, cfg)
    chain.model.print_trainable_parameters()
    lora = [p for p in chain.model.parameters() if p.requires_grad]
    groups = [{"params": lora, "lr": a.lr_lora}, {"params": list(chain.head.parameters()), "lr": 1e-4},
              {"params": list(chain.proj.parameters()) + [chain.img_start], "lr": 3e-4}]
    banner("9a: eval before"); ev0 = evaluate(chain, val, use_amp); show("before", ev0)
    banner("9b: training"); train_loop(chain, groups, train, val, a.epochs, a.bs, use_amp, "lora")
    banner("9c: summary"); ev1 = evaluate(chain, val, use_amp); show("after ", ev1)
    print(f"\nLM loss (feedback) {ev0['lm_feedback']:.4f} -> {ev1['lm_feedback']:.4f} | imagination_benefit {ev0['imagination_benefit']:.4f} -> {ev1['imagination_benefit']:.4f}")
    chain.model.save_pretrained(f"{CKPT}/lora_step9"); save_chain_parts(chain, f"{CKPT}/chain_step9.pt")
    print(f"saved {CKPT}/lora_step9 and chain_step9.pt\nSTEP 9 OK")

if __name__ == "__main__":
    main()
