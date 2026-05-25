import json
import numpy as np
import matplotlib.pyplot as plt

with open("lambda_sweep_results.json") as f:
    res = json.load(f)

lams = sorted([float(k) for k in res.keys()])
acc_mean = [res[str(l)]["acc_mean"] for l in lams]
acc_std  = [res[str(l)]["acc_std"] for l in lams]
ent_mean = [res[str(l)]["entropy_mean"] for l in lams]
ent_std  = [res[str(l)]["entropy_std"] for l in lams]

plt.figure(figsize=(10,4))

# Accuracy vs lambda
plt.subplot(1,2,1)
plt.errorbar(lams, acc_mean, yerr=acc_std, marker="o")
plt.xscale("log")
plt.xlabel("λ (salience supervision)")
plt.ylabel("Accuracy @2048")
plt.title("Accuracy vs λ")

# Entropy vs lambda
plt.subplot(1,2,2)
plt.errorbar(lams, ent_mean, yerr=ent_std, marker="o", color="orange")
plt.xscale("log")
plt.xlabel("λ (salience supervision)")
plt.ylabel("Write entropy")
plt.title("Memory Entropy vs λ")

plt.tight_layout()
plt.show()
