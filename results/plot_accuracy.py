import matplotlib.pyplot as plt

# example accuracy values from experiment
rounds = [1, 2, 3]

accuracy = [1.0, 0.617, 1.0]

plt.figure(figsize=(6,4))

plt.plot(rounds, accuracy, marker='o', linewidth=2)

plt.title("Federated IDS Performance under Adversarial Attack")

plt.xlabel("Federated Training Round")

plt.ylabel("Detection Accuracy")

plt.grid(True)

plt.ylim(0,1.1)

plt.savefig("results/accuracy_graph.png")

plt.show()