import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics_log.csv")

rounds = df["round"]

# Accuracy
plt.figure()
plt.plot(rounds, df["accuracy"], marker="o")
plt.title("Federated Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("results/round_accuracy.png")

# Precision
plt.figure()
plt.plot(rounds, df["precision"], marker="o")
plt.title("Precision vs Rounds")
plt.xlabel("Round")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("results/precision_rounds.png")

# Recall
plt.figure()
plt.plot(rounds, df["recall"], marker="o")
plt.title("Recall vs Rounds")
plt.xlabel("Round")
plt.ylabel("Recall")
plt.grid(True)
plt.savefig("results/recall_rounds.png")

# F1 Score
plt.figure()
plt.plot(rounds, df["f1_score"], marker="o")
plt.title("F1 Score vs Rounds")
plt.xlabel("Round")
plt.ylabel("F1 Score")
plt.grid(True)
plt.savefig("results/f1_rounds.png")

# ROC AUC
plt.figure()
plt.plot(rounds, df["roc_auc1"], marker="o")
plt.title("ROC AUC vs Rounds")
plt.xlabel("Round")
plt.ylabel("ROC AUC")
plt.grid(True)
plt.savefig("results/roc_auc_rounds.png")

print("All graphs generated successfully.")