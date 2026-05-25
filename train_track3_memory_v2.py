import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from synapnet_memory import SynapEpisodicNet
import random

def make_sample(seq_len=2048, vocab_tokens=2048, codebook_size=32):
    fact_id = random.randint(0, codebook_size - 1)
    x = torch.randint(0, vocab_tokens, (seq_len,))
    x[0] = 10
    x[1] = 11
    x[2] = fact_id
    x[-3] = 12
    x[-2] = 13
    x[-1] = 14
    label = fact_id
    return x, label

class MemoryDatasetV2(Dataset):
    def __init__(self, n=200, seq_len=2048, vocab_tokens=2048, codebook_size=32):
        self.samples = [make_sample(seq_len, vocab_tokens, codebook_size) for _ in range(n)]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]

def train_memory_v2(model, loader, device="cuda", epochs=10):
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        total_loss, total_acc = 0.0, 0.0
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, debug_masks, debug_mems, debug_indices = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            total_acc += (pred == yb).float().mean().item()
            if batch_idx == 0:
                first_indices = []
                for layer_idx, slot_idx in enumerate(debug_indices):
                    idx_list = [int(i) for i in slot_idx[0].detach().cpu().tolist() if i >= 0]
                    first_indices.append((layer_idx, idx_list))
                printable = ", ".join(f"block{layer}: {idxs}" for layer, idxs in first_indices)
                print(f"  [Epoch {epoch+1}] stored timesteps -> {printable}")
        print(f"[Epoch {epoch+1}] loss={total_loss/len(loader):.4f} acc={total_acc/len(loader):.4f}")
    torch.save(model.state_dict(), "synapnet_memory_v2.pt")

if __name__ == "__main__":
    ds = MemoryDatasetV2(n=128, seq_len=2048, vocab_tokens=2048, codebook_size=32)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SynapEpisodicNet(dim=128,
                             depth=3,
                             vocab_size=2048,
                             output_size=32,
                             max_len=4096,
                             heads=4,
                             k_frac=0.25,
                             episodic_slots=8,
                             episodic_write_frac=0.05)
    train_memory_v2(model, dl, device=device, epochs=10)
