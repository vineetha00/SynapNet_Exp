# run_comparisons.py
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from baselines import TransformerBaseline, SSMProxy
from train_track3_memory_v4 import MemoryDatasetV4, SynapEpisodicNet, train_generic  # import SynapEpisodicNet from your file

SEEDS = [0, 1, 2]
BATCH_SIZE = 4
EPOCHS = 10

def run_model_factory(factory_fn, is_synap=False, lambda_mem=0.0):
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        ds = MemoryDatasetV4(n=256)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        model = factory_fn()
        m = train_generic(model, dl, device="cuda" if torch.cuda.is_available() else "cpu",
                          epochs=EPOCHS, lambda_mem=lambda_mem, is_synap=is_synap, seed=seed)
        results.append(m)
    # aggregate
    accs = [r["acc"] for r in results]
    avg_acc = sum(accs)/len(accs)
    hit_rates = [r["hit_rate"] for r in results if r["hit_rate"] is not None]
    avg_hit = (sum(hit_rates)/len(hit_rates)) if len(hit_rates)>0 else None
    return {"avg_acc": avg_acc, "avg_hit": avg_hit, "per_run": results}

def main():
    out = {}

    # Transformer baseline
    out["transformer"] = run_model_factory(lambda: TransformerBaseline(dim=128, depth=3, vocab_size=2048, max_len=2049, heads=4, num_classes=32), is_synap=False)

    # SSM proxy baseline
    out["ssmproxy"] = run_model_factory(lambda: SSMProxy(dim=128, depth=3, vocab_size=2048, num_classes=32), is_synap=False)

    # SynapNet no supervision
    out["synap_lambda0"] = run_model_factory(lambda: SynapEpisodicNet(dim=128, depth=3, vocab_size=2048, max_len=2049, heads=4, k_frac=0.25, episodic_slots=8, episodic_write_frac=0.05, num_classes=32), is_synap=True, lambda_mem=0.0)

    # SynapNet weak supervision
    out["synap_lambda01"] = run_model_factory(lambda: SynapEpisodicNet(dim=128, depth=3, vocab_size=2048, max_len=2049, heads=4, k_frac=0.25, episodic_slots=8, episodic_write_frac=0.05, num_classes=32), is_synap=True, lambda_mem=0.1)

    # SynapNet strong KL supervision
    out["synap_lambda1"] = run_model_factory(lambda: SynapEpisodicNet(dim=128, depth=3, vocab_size=2048, max_len=2049, heads=4, k_frac=0.25, episodic_slots=8, episodic_write_frac=0.05, num_classes=32), is_synap=True, lambda_mem=1.0)

    # save results
    import json
    with open("comparison_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("Done. Results saved to comparison_results.json")
    print(out)

if __name__ == "__main__":
    main()
