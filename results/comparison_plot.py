import matplotlib.pyplot as plt

# experimental results (example values from your run)
methods = [
    "Normal FL",
    "Under Attack",
    "Median Defense",
    "Trimmed Mean",
    "Krum"
]

accuracy=[
    1.00,
    0.61,
    1.00,
    0.97,
    0.95
]

plt.figure(figsize=(8,5))

bars = plt.bar(methods, accuracy)

plt.title("Attack vs Defense Performance in Federated IDS")

plt.xlabel("Method")

plt.ylabel("Detection Accuracy")

plt.ylim(0,1.1)

plt.grid(axis='y')

plt.savefig("results/attack_vs_defese.png")

plt.show()

"""
Slide Title:

Normal federated learning achieved high accuracy.
however, adversarial clients injecting poisoned updates reduced model performance significantly.

Robust aggregation techniques such as Median Trimmed Mean, and Krum successfully mitigated the attack.
"""