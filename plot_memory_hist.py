# plot_memory_hist.py
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(npz_path, title, highlight_idx=2, top_k=20):
    data = np.load(npz_path)
    indices = data["indices"]
    freqs = data["freqs"]

    order = np.argsort(freqs)[-top_k:]
    idx = indices[order]
    fr = freqs[order]

    plt.figure(figsize=(7, 4))
    plt.bar(idx, fr)
    plt.axvline(highlight_idx, color="red", linestyle="--", label="Target index")
    plt.xlabel("Token index")
    plt.ylabel("Normalized write frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- RUN ----
plot_hist(
    "write_hist_lambda_0.0.npz",
    title="Memory Write Distribution (λ = 0.0)"
)

"""plot_hist(
    "write_hist_lambda_1.0.npz",
    title="Memory Write Distribution (λ = 1.0, KL Supervision)"
)"""
