import json
import matplotlib.pyplot as plt

with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

r2 = metrics["R2_score"]
mse = metrics["MSE"]

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_title("Linear Regression Metrics (Original Model)")

# R² bar
bars1 = ax1.bar(["R² Score"], [r2], color="skyblue")
ax1.set_ylim(0, 1)
ax1.set_ylabel("R² Score", color="skyblue")

# Add value label
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# MSE bar
ax2 = ax1.twinx()
bars2 = ax2.bar(["MSE"], [mse], color="salmon", width=0.4)
ax2.set_ylim(0, 2)
ax2.set_ylabel("MSE", color="salmon")

# Add value label
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("models/metrics_plot.png")
print("metrics_plot.png generated and saved.")
