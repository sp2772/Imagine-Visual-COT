# Step 2: Imagination Head (z_t = W2 h_t) and Gate (g_t = sigmoid(W1 h_t)).
# A LayerNorm sits in front of both linear maps for stable training (remove it for the exact paper formula).
import torch, torch.nn as nn, torch.nn.functional as F

D_VIS = 512  # CLIP ViT-B/32 embedding size

class ImaginationHead(nn.Module):
    def __init__(self, d_model, d_vis=D_VIS):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.W2 = nn.Linear(d_model, d_vis)   # imagination vector
        self.W1 = nn.Linear(d_model, 1)       # gate

    def forward(self, h):
        x = self.norm(h)
        z = F.normalize(self.W2(x), dim=-1)           # unit norm -> cosine similarity is meaningful
        g = torch.sigmoid(self.W1(x)).squeeze(-1)
        return z, g

if __name__ == "__main__":
    from imagine_step1 import load_backbone, get_hidden, device
    tok, model = load_backbone()
    head = ImaginationHead(model.config.hidden_size).to(device)

    h = get_hidden(tok, model, "Step 1: the box is 10 cm wide and 8 cm tall.")
    z, g = head(h)
    print("h:", tuple(h.shape), "-> z:", tuple(z.shape), "g:", tuple(g.shape), "(expect z [1,L,512], g [1,L])")
    print("z norms (should be 1.0):", [round(x, 3) for x in z.norm(dim=-1)[0, :3].tolist()])
    print("gate values (random init, ~0.5):", [round(x, 3) for x in g[0, :5].tolist()])
    print("trainable params in head:", sum(p.numel() for p in head.parameters()))
    z2, _ = head(h[:, -1])
    print("single-vector call: z", tuple(z2.shape), "(expect [1,512])")
    print("\nSTEP 2 OK")
