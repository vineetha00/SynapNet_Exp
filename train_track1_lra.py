# TRACK 1: Long-Range Sequence Modeling / LRA-style benchmark
# Skeleton training loop comparing SynapNet vs Transformer on long sequences.
# NOTE: You will need to plug in a dataset loader that yields (input_ids, labels)
# for long-context classification tasks.

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from synapnet import SynapNet

###############################################
# Dummy dataset scaffold for long sequences
###############################################
class LongSeqDataset(Dataset):
    def __init__(self, num_samples=200, seq_len=4096, vocab_size=1000, num_classes=10):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i], self.labels[i]

###############################################
# Classification head mode (num_classes != None)
###############################################
class SynapNetClassifier(nn.Module):
    def __init__(self, num_classes, dim=128, depth=4, vocab_size=1000, max_len=8192):
        super().__init__()
        self.model = SynapNet(dim=dim,
                              depth=depth,
                              vocab_size=vocab_size,
                              max_len=max_len,
                              num_classes=num_classes)
    def forward(self, idx):
        logits, masks = self.model(idx)  # (B,C), list[(B,T)]
        return logits, masks

###############################################
# Training loop
###############################################
def train_cls(model, loader, device="cuda"):
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(3):
        total_loss, total_acc = 0.0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, masks = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            total_acc += (pred == yb).float().mean().item()
        print(f"[Epoch {epoch+1}] loss={total_loss/len(loader):.4f} acc={total_acc/len(loader):.4f}")

if __name__ == "__main__":
    ds = LongSeqDataset(num_samples=64, seq_len=2048, vocab_size=1000, num_classes=5)
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    model = SynapNetClassifier(num_classes=5, dim=64, depth=3, vocab_size=1000, max_len=4096)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cls(model, dl, device=device)
