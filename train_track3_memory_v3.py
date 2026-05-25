import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from synapnet_memory import SynapEpisodicNet
import random

# ============================
# CONFIG
# ============================
SEQ_LEN = 2048          # length of sequence
VOCAB_SIZE = 2048       # distractor token space
NUM_CODES = 32          # number of possible "secret codes"
EPOCHS = 10
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# DATASET
# ============================
def make_sample(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_codes=NUM_CODES):
    """
    Build a long sequence:
      tokens[0:3] = "the code is <CODE>"
      tokens[-3:] = "what is the code ?"
    The label is that code (0..num_codes-1).
    """
    # choose a code from a SMALL set, 0..num_codes-1
    code_id = random.randint(0, num_codes - 1)

    # draw distractor tokens from a large vocab
    x = torch.randint(0, vocab_size, (seq_len,))

    # plant the code up front
    # 10,11 are like "the code is"
    x[0] = 10
    x[1] = 11
    # store code_id in position 2
    # NOTE: we store code_id itself (0..31). This lets the model learn that
    # that position is special.
    x[2] = code_id

    # ask at the end (12,13,14 are like "what is code?")
    x[-3] = 12
    x[-2] = 13
    x[-1] = 14

    label = code_id  # supervision is just which code it was
    return x, label


class MemoryDatasetV3(Dataset):
    def __init__(self, n=256):
        self.samples = [make_sample() for _ in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ============================
# TRAIN LOOP
# ============================
def train_memory_v3(model, loader, device=DEVICE):
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_acc = 0.0

        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits, debug_masks, debug_mems, debug_topk = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                batch_acc = (pred == yb).float().mean().item()

            total_loss += loss.item()
            total_acc += batch_acc

            # DEBUG PRINT (first batch of first epoch or so)
            if epoch == 0 and step == 0:
                # look at which indices got written into memory, block 0
                first_block_topk = debug_topk[0][0].detach().cpu().tolist()
                wrote_code_token = (2 in first_block_topk)

                print("---- DEBUG SAMPLE ----")
                print("label (true code_id):", yb[0].item())
                print("predicted:", pred[0].item())
                print("topk write indices (block0):", first_block_topk[:10])
                print("did it write index 2 (the code)?", wrote_code_token)
                print("----------------------")

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} acc={avg_acc:.4f}")

    torch.save(model.state_dict(), "synapnet_memory_v3.pt")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    ds = MemoryDatasetV3(n=256)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SynapEpisodicNet(
        dim=128,
        depth=3,
        vocab_size=VOCAB_SIZE,
        max_len=SEQ_LEN + 1,
        heads=4,
        k_frac=0.25,
        episodic_slots=8,
        episodic_write_frac=0.05,
        num_classes=NUM_CODES,
    )

    train_memory_v3(model, dl, device=DEVICE)
