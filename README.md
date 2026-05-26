# SynapNet — Hybrid SSM + Sparse Attention + Episodic Memory

Research codebase for the base **SynapNet** architecture: a hybrid sequence model that combines a depthwise-conv SSM branch, a salience-gated SparseEventAttention, and a writeable episodic memory with learned fusion gates.

🤗 **Checkpoints:** https://huggingface.co/Vineetha00/synapnet
🛠️ **Deployment companion:** [SynapNet-Edge](https://github.com/vineetha00/SynapNet-Edge) · 🤗 https://huggingface.co/Vineetha00/synapnet-edge

This is the architectural foundation. The edge-deployment systems work (component-aware quantization, budget-aware eviction, hardware benchmarking) lives in the companion repository **[SynapNet-Edge](https://github.com/vineetha00/SynapNet-Edge)**.

---

## What's in here

| File | Purpose |
|---|---|
| [`synapnet.py`](synapnet.py) | Base architecture: `SimpleSSM`, `SparseEventAttention`, `ExternalMemory`, `SynapBlock`, `SynapNet` |
| [`synapnet_memory.py`](synapnet_memory.py) | Episodic-memory variant: `WriteableMemory`, `SynapBlockWithEpisodic`, `SynapEpisodicNet` |
| [`baselines.py`](baselines.py) | Transformer + SSM-proxy baselines for head-to-head comparisons |
| [`train_track1_lra.py`](train_track1_lra.py) | Long-range sequence classification (LRA-style) |
| [`train_track2_biosignal.py`](train_track2_biosignal.py) | Streaming biosignal regression (ECG-style) |
| [`train_track3_memory_v2.py`](train_track3_memory_v2.py) … `_v4.py` | Episodic-recall task (secret-code retrieval across long gaps) |
| [`train_track4_interpret.py`](train_track4_interpret.py) | Faithful-salience supervision for interpretability |
| [`run_comparisons.py`](run_comparisons.py) | Multi-seed Transformer / SSM / SynapNet comparison |
| [`run_lambda_sweep.py`](run_lambda_sweep.py) | Salience-supervision strength sweep |
| `plot_*.py` | Result plotters |
| `synapnet_memory*.pt` | Pretrained checkpoints for the episodic variants |
| `comparison_results.json`, `lambda_sweep_results.json` | Multi-seed experiment outputs |

---

## Architecture (one block)

```
x ─┬─ SimpleSSM (depthwise conv + gate)              ─┐
   ├─ SparseEventAttention → soft salience mask      ─┤── α·attn + β·mem
   └─ ExternalMemory  /  WriteableMemory (top-K)     ─┘
              │
              ▼
       residual + FFN → next block
```

`SparseEventAttention` produces a per-token salience score; `WriteableMemory` uses that score to write the top-K hidden states into a small fixed bank, which later tokens read via cross-attention.

---

## Key experimental finding (Track 3, episodic recall)

From `lambda_sweep_results.json` — salience-supervision strength λ controls a clean trade-off between accuracy, write-hit-rate (does the model remember the right token?), and salience entropy. Mean ± std over 5 seeds, ctx = 2048:

| λ | accuracy | write hit-rate |
|---|---|---|
| 0.00 | 0.723 ± 0.329 | 0.125 ± 0.109 |
| **0.01** | **0.968 ± 0.022** | **0.993 ± 0.002** |
| 0.10 | 0.970 ± 0.021 | 0.995 ± 0.001 |
| 1.00 | 0.838 ± 0.125 | 0.995 ± 0.002 |

A small dose of salience supervision (λ = 0.01) is enough to flip the model from chance-level recall to near-perfect.

---

## Quickstart

```bash
git clone https://github.com/vineetha00/SynapNet_Exp
cd SynapNet_Exp
pip install torch numpy matplotlib pandas

# Track 1 — long-range classification
python train_track1_lra.py

# Track 2 — biosignal regression (place ecg_sample.csv under data/)
python train_track2_biosignal.py

# Track 3 — episodic recall (v4 is the current best)
python train_track3_memory_v4.py

# Track 4 — faithful salience supervision
python train_track4_interpret.py

# Multi-seed comparison + lambda sweep
python run_comparisons.py
python run_lambda_sweep.py
```

---

## Citation

```bibtex
@article{synapnet_2026,
  title={SynapNet: Hybrid SSM + Sparse-Attention + Episodic Memory for Long-Range Sequence Modelling},
  author={Vallish Kumar, Vineetha},
  year={2026},
}
```

## Related work

**Deployment / efficiency:** [SynapNet-Edge](https://github.com/vineetha00/SynapNet-Edge) — component-aware quantization (CAJQ), budget-aware episodic eviction (BAEE), and consumer-hardware benchmarks built on this architecture.

## License

MIT
