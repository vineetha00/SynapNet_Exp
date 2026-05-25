import numpy as np
import matplotlib.pyplot as plt

T = 2048
ts = np.random.normal(0, 0.1, T)

# Early pattern (pattern A)
pattern_A = np.sin(np.linspace(0, 4*np.pi, 40))
ts[50:90] += pattern_A

# Late pattern (either matching or different)
pattern_B = np.sin(np.linspace(0, 4*np.pi, 40))
ts[1800:1840] += pattern_B  # matching case

plt.figure(figsize=(10,4))
plt.plot(ts)
plt.axvspan(50, 90, color='red', alpha=0.2, label='Early pattern')
plt.axvspan(1800, 1840, color='green', alpha=0.2, label='Late pattern')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Long-Range Dependency Example (2048 steps)")
plt.legend()
plt.tight_layout()
plt.show()
