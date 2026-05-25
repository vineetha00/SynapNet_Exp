# TRACK 4: Interpretability / Faithful Salience
# Synthetic text classification with known trigger spans.
# We supervise attention masks to focus on the true trigger tokens.

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from synapnet import SynapNet

TRIGGER_TOKEN = 42  # stand-in for the "clinically relevant" phrase

def make_trigger_sample(seq_len=128, vocab_size=500, trigger_prob=0.5):
    x = torch.randint(0, vocab_size, (seq_len,))
    rationale_mask = torch.zeros(seq_len)
    if torch.rand(1).item() < trigger_prob:
        pos = torch.randint(low=10, high=seq_len-10, size=(1,)).item()
        x[pos] = TRIGGER_TOKEN
        label = 1
        rationale_mask[pos] = 1.0
    else:
        label = 0
    return x, label, rationale_mask

class RationaleDataset(Dataset):
    def __init__(self, n=256, seq_len=128, vocab_size=500):
        self.samples = [make_trigger_sample(seq_len, vocab_size) for _ in range(n)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]  # (x,label,mask)

class SynapNetClassifier(nn.Module):
    def __init__(self, num_classes, dim=128, depth=4, vocab_size=500, max_len=512):
        super().__init__()
        self.model = SynapNet(dim=dim,
                              depth=depth,
                              vocab_size=vocab_size,
                              max_len=max_len,
                              num_classes=num_classes)
    def forward(self, idx):
        logits, masks = self.model(idx)  # logits: (B,C), masks: list[(B,T)]
        return logits, masks

def train_with_rationale(model, loader, device="cuda", lambda_mask=0.1):
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(5):
        total_cls, total_maskloss, total_acc = 0.0, 0.0, 0.0
        for xb, yb, rb in loader:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            opt.zero_grad()
            logits, masks = model(xb)   # masks: list of salience maps per block
            cls_loss = ce(logits, yb)

            # Take first block's salience as explanation
            expl = masks[0]            # (B,T)
            # BCE between expl and rb (target rationale mask)
            mask_loss = F.binary_cross_entropy(
                torch.clamp(expl, 1e-6, 1-1e-6),
                rb
            )

            loss = cls_loss + lambda_mask * mask_loss
            loss.backward()
            opt.step()

            total_cls += cls_loss.item()
            total_maskloss += mask_loss.item()
            pred = logits.argmax(dim=-1)
            total_acc += (pred == yb).float().mean().item()

        print(f"[Epoch {epoch+1}] acc={total_acc/len(loader):.4f}  "
              f"CE={total_cls/len(loader):.4f}  "
              f"maskLoss={total_maskloss/len(loader):.4f}")
    torch.save(model.state_dict(), "synapnet_interpret.pt")

if __name__ == "__main__":
    ds = RationaleDataset(n=256, seq_len=128, vocab_size=500)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SynapNetClassifier(num_classes=2, dim=64, depth=3, vocab_size=500, max_len=256)
    train_with_rationale(model, dl, device=device, lambda_mask=0.2)
