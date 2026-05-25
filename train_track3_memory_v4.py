import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from synapnet_memory import SynapEpisodicNet
import random
from collections import Counter
import torch.nn.functional as F
import numpy as np


# ============================
# CONFIG
# ============================
SEQ_LEN = 2048          # tokens per sequence
VOCAB_SIZE = 2048       # distractor token space
NUM_CODES = 32          # number of possible secret codes (classes)
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_MEM = 0.0       # weight on salience supervision


# ============================
# DATASET
# ============================
def make_sample(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, num_codes=NUM_CODES):
    """
    Build a long sequence:
      tokens[0:3] ~ "the code is <CODE>"
      tokens[-3:] ~ "what is the code?"
    The label = <CODE> from 0..num_codes-1.

    Important detail:
    - We ALWAYS store the code_id itself at position 2.
    - We'll teach the model that position 2 is "salient" and must be written.
    """
    code_id = random.randint(0, num_codes - 1)

    # distractor tokens
    x = torch.randint(0, vocab_size, (seq_len,))

    # plant the code early
    x[0] = 10        # "the"
    x[1] = 11        # "code_is"
    x[2] = code_id   # <-- THIS is the fact we want it to remember

    # ask about it at the end
    x[-3] = 12       # "what"
    x[-2] = 13       # "is_the_code?"
    x[-1] = 14       # "answer_here"

    label = code_id
    return x, label


class MemoryDatasetV4(Dataset):
    def __init__(self, n=256):
        self.samples = [make_sample() for _ in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
    
def train_generic(
    model,
    loader,
    device="cpu",
    epochs=10,
    lambda_mem=0.0,
    is_synap=False,
    seed=0,
):
    torch.manual_seed(seed)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    write_counter = Counter()
    total_hit = 0
    total_samples = 0

    for epoch in range(epochs):
        model.train()
        total_acc = 0.0

        for batch in loader:
            if len(batch) == 3:
                xb, yb, code_pos = batch
                code_pos = code_pos.to(device)
            else:
                xb, yb = batch
                code_pos = None

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits, debug_masks, debug_mems, debug_topk = model(xb)

            cls_loss = ce(logits, yb)
            loss = cls_loss

            if is_synap and lambda_mem > 0.0:
                B, T = xb.shape
                target_write_mask = torch.zeros((B, T), device=device)

                if code_pos is None:
                    target_write_mask[:, 2] = 1.0
                else:
                    target_write_mask[torch.arange(B), code_pos] = 1.0

                target_write_mask /= target_write_mask.sum(dim=1, keepdim=True)

                sal_pred_block0 = debug_masks[0]
                mem_loss = F.kl_div(
                    sal_pred_block0.log_softmax(dim=-1),
                    target_write_mask,
                    reduction="batchmean",
                )

                loss = cls_loss + lambda_mem * mem_loss

            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                total_acc += (pred == yb).float().mean().item()

                if is_synap and debug_topk is not None:
                    topk0 = debug_topk[0]
                    for i in range(topk0.size(0)):
                        tk = topk0[i].cpu().tolist()
                        for idx in tk:
                            write_counter[idx] += 1

                        if code_pos is None:
                            hit = 2 in tk
                        else:
                            hit = code_pos[i].item() in tk

                        total_hit += int(hit)
                        total_samples += 1

        avg_acc = total_acc / len(loader)
        print(f"[Epoch {epoch+1}] acc={avg_acc:.4f}")

    hit_rate = total_hit / total_samples if total_samples > 0 else None
    return {
        "acc": avg_acc,
        "hit_rate": hit_rate,
        "write_counter": write_counter,
    }


# ============================
# TRAIN LOOP
# ============================
def train_memory_v4(model, loader, device=DEVICE):
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    model.to(device)

    for epoch in range(EPOCHS):
        write_counter = Counter()
        total_samples = 0
        hit_count = 0
        total_loss = 0.0
        total_cls_loss = 0.0
        total_mem_loss = 0.0
        total_acc = 0.0

        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()

            logits, debug_masks, debug_mems, debug_topk = model(xb)
            # debug_topk: list over blocks -> [B, K]
            topk_block0 = debug_topk[0]  # shape: [B, K]

            for i in range(topk_block0.size(0)):
                indices = topk_block0[i].detach().cpu().tolist()
                total_samples += 1

                # hit rate
                if 2 in indices:
                    hit_count += 1

                # histogram
                for idx in indices:
                    write_counter[idx] += 1

            cls_loss = ce(logits, yb)

            B, T = xb.shape
            target_write_mask = torch.zeros((B, T), device=device, dtype=torch.float32)
            target_write_mask[:] = 0.0
            target_write_mask[:, 2] = 1.0
            target_write_mask /= target_write_mask.sum(dim=1, keepdim=True)


            sal_pred_block0 = debug_masks[0]
            mem_loss = F.kl_div(
                sal_pred_block0.log_softmax(dim=-1),
                target_write_mask,
                reduction="batchmean"
            )


            loss = cls_loss + LAMBDA_MEM * mem_loss
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                batch_acc = (pred == yb).float().mean().item()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mem_loss += mem_loss.item()
            total_acc += batch_acc

            if epoch == EPOCHS - 1:
                hit_rate = hit_count / total_samples
                print("\n==== MEMORY AUDIT @ FINAL EPOCH ====")
                print(f"Hit rate (index 2 in top-k): {hit_rate:.3f}")

                print("\nTop 10 most written indices:")
                for idx, cnt in write_counter.most_common(10):
                    print(f"Index {idx}: {cnt}")

                print("===================================\n")


        avg_loss = total_loss / len(loader)
        avg_cls_loss = total_cls_loss / len(loader)
        avg_mem_loss = total_mem_loss / len(loader)
        avg_acc = total_acc / len(loader)

        print(f"[Epoch {epoch+1}] "
              f"total_loss={avg_loss:.4f} "
              f"cls_loss={avg_cls_loss:.4f} "
              f"mem_loss={avg_mem_loss:.4f} "
              f"acc={avg_acc:.4f}")

    torch.save(model.state_dict(), "synapnet_memory_v4.pt")

    total_writes = sum(write_counter.values())
    print("\nNormalized top-10 write frequencies:")
    for idx, cnt in write_counter.most_common(10):
        print(f"Index {idx}: {cnt / total_writes:.3f}")


    import numpy as np

    total = sum(write_counter.values())
    indices = []
    freqs = []

    for idx, cnt in write_counter.items():
        indices.append(idx)
        freqs.append(cnt / total)

    np.savez(
        f"write_hist_lambda_{LAMBDA_MEM}.npz",
        indices=np.array(indices),
        freqs=np.array(freqs)
    )

if __name__ == "__main__":
    ds = MemoryDatasetV4(n=256)
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

    train_memory_v4(model, dl, device=DEVICE)

    


        
        

