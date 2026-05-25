# plot_results.py
import json
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("comparison_results.json"))

models = ["transformer", "ssmproxy", "synap_lambda0", "synap_lambda01", "synap_lambda1"]
names = ["Transformer", "SSMProxy", "Synap (λ=0)", "Synap (λ=0.1)", "Synap (λ=1.0 KL)"]
accs = [data[m]["avg_acc"] for m in models]
hits = [data[m]["avg_hit"] if data[m]["avg_hit"] is not None else 0 for m in models]

# accuracy bar
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.bar(names, accs)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Accuracy @2048")

# hit rate (only for synap)
plt.subplot(1,2,2)
plt.bar(names, hits)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Avg Hit Rate (index in top-k)")
plt.ylim(0,1.05)
plt.tight_layout()
plt.show()
