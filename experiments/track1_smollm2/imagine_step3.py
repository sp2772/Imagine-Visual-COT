# Step 3: anti-collapse loss. Simulate a COLLAPSED head, then fix it with a diversity loss.
import torch, torch.nn.functional as F
from tqdm import tqdm
from imagine_step1 import load_backbone, get_hidden, device
from imagine_step2 import ImaginationHead

STEPS = [
    "Question: Can a ball of 9 cm diameter fit in a box 10 cm wide and 8 cm tall?",
    "Step 1: The box is 10 cm wide and 8 cm tall.",
    "Step 2: The ball needs 9 cm of room in every direction.",
    "Step 3: 9 cm is less than 10 cm, so it fits in width.",
    "Step 4: 9 cm is more than 8 cm, so it does not fit in height.",
    "Answer: No, the ball cannot fit.",
]

def step_hiddens(tok, model):
    """h_t for each reasoning step = hidden state at the last token of the text so far."""
    hs, text = [], ""
    for s in STEPS:
        text += s + "\n"
        hs.append(get_hidden(tok, model, text)[0, -1])
    return torch.stack(hs)                                  # [T, d_model]

def consecutive_cos(z):
    return F.cosine_similarity(z[1:], z[:-1], dim=-1)       # [T-1]

def diversity_loss(z, margin=0.3):
    # penalise z_t that is too similar to z_{t-1}
    return F.relu(consecutive_cos(z) - margin).mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    tok, model = load_backbone()
    H = step_hiddens(tok, model)
    head = ImaginationHead(model.config.hidden_size).to(device)

    # force collapse: a huge constant bias dominates, so every z_t looks the same
    with torch.no_grad():
        head.W2.bias.copy_(torch.randn_like(head.W2.bias) * 50)

    z, _ = head(H)
    print("BEFORE  mean cos(z_t, z_t-1): %.3f  (collapsed ~1.0)" % consecutive_cos(z).mean())

    opt = torch.optim.Adam(head.parameters(), lr=1e-1)
    pbar = tqdm(range(300), desc="anti-collapse training")
    for i in pbar:
        z, _ = head(H)
        loss = diversity_loss(z)
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{consecutive_cos(z).mean().item():.3f}")

    z, _ = head(H)
    print("AFTER   mean cos(z_t, z_t-1): %.3f  (should be <= ~0.3)" % consecutive_cos(z).mean())
    print("\nSTEP 3 OK")
