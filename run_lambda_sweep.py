import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from train_track3_memory_v4 import MemoryDatasetV4, train_generic
from synapnet_memory import SynapEpisodicNet
from baselines import SSMProxy

def entropy_from_counter(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()])
    return -np.sum(probs * np.log(probs + 1e-12))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 10

SEEDS = [0, 1, 2, 3, 4]   # start with 5, you can extend to 10
LAMBDAS = [0.0, 0.01, 0.1, 1.0]  # for plots later

results = {}

for lam in LAMBDAS:
    accs = []
    hits = []
    ents = []

    print(f"\n=== SynapNet λ = {lam} ===")

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ds = MemoryDatasetV4(n=256)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        model = SynapEpisodicNet(
            dim=128,
            depth=3,
            vocab_size=2048,
            max_len=2049,
            heads=4,
            k_frac=0.25,
            episodic_slots=8,
            episodic_write_frac=0.05,
            num_classes=32,
        )

        out = train_generic(
            model,
            dl,
            device=DEVICE,
            epochs=EPOCHS,
            lambda_mem=lam,
            is_synap=True,
            seed=seed,
        )

        accs.append(out["acc"])
        hits.append(out["hit_rate"])
        ents.append(entropy_from_counter(out["write_counter"]))

    results[lam] = {
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "hit_mean": np.mean(hits),
        "hit_std": np.std(hits),
        "entropy_mean": np.mean(ents),
        "entropy_std": np.std(ents),
    }

    print(results[lam])

    print("\n=== SSMProxy reference ===")

ssm_accs = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ds = MemoryDatasetV4(n=256)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SSMProxy(
        dim=128,
        depth=3,
        vocab_size=2048,
        num_classes=32,
    )

    out = train_generic(
        model,
        dl,
        device=DEVICE,
        epochs=EPOCHS,
        is_synap=False,
        seed=seed,
    )

    ssm_accs.append(out["acc"])

print("SSMProxy mean ± std:",
      np.mean(ssm_accs), "±", np.std(ssm_accs))


import json
with open("lambda_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)



