Memory V2 Experiment
====================

Goal:
-----
Give the network explicit episodic memory:
  1. Detect salient tokens via SparseEventAttention.
  2. Write those hidden states into a per-sequence memory bank.
  3. Read that bank at the end to answer a question about the past.

Files:
------
synapnet_memory.py
    - WriteableMemory
    - SynapBlockWithEpisodic
    - SynapEpisodicNet

train_track3_memory_v2.py
    - Builds synthetic sequences where a "secret code" appears at the start,
      and is queried at the end.
    - Trains the model to output that code ID.

Usage:
------
python3 train_track3_memory_v2.py

What you should see:
--------------------
Accuracy should climb noticeably above ~0 after a few epochs.
That's the model storing the early fact in episodic memory and
retrieving it, instead of relying on raw long-range attention.

This is the concrete gap vs. Transformers:
- Transformers have no explicit writeable episodic memory path.
- SynapEpisodicNet does, and the task is built to expose that.
