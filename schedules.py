import numpy as np
import matplotlib.pyplot as plt

T = 1000
steps = np.arange(T)

t = steps / (T - 1)

base_schedules = {
    "Linear": t,
    "Logarithmic": np.log1p(9 * t) / np.log(10),
    "Exponential": (np.exp(5 * t) - 1) / (np.exp(5) - 1),
    "Cosine": 0.5 * (1 - np.cos(np.pi * t)),
}

beta_schedules = {
    name: 1.0 + 2.0 * sched
    for name, sched in base_schedules.items()
}

colors = {
    "Linear": "tab:blue",
    "Logarithmic": "tab:orange",
    "Exponential": "tab:green",
    "Cosine": "tab:red",
}

plt.figure(figsize=(8, 5))

for label, beta in beta_schedules.items():
    plt.plot(
        steps,
        beta,
        linewidth=2.5,
        label=label,
        color=colors[label],
    )

plt.legend(fontsize=12, framealpha=0.9, loc="best")

plt.xlabel("Step", fontsize=20)
plt.ylabel(r"Schedule value", fontsize=20)
plt.title(r"Comparison of scheduling functions",
          fontsize=18, fontweight="bold")

plt.xlim(left=0)
plt.ylim(0.95, 3.05)

plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
plt.savefig("figures/beta_schedules.png", dpi=150, bbox_inches="tight")

plt.show()