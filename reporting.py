import json
import matplotlib.pyplot as plt

with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

r2 = metrics["R2_score"]
mse = metrics["MSE"]

fig, ax1 = plt.subplots()

ax1.set_title("Model Metrics")
ax1.bar(["R² Score"], [r2], color="skyblue")
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.bar(["MSE"], [mse], color="salmon", width=0.4)
ax2.set_ylim(0, 2)

plt.savefig("models/metrics_plot.png")
print("metrics_plot.png generated.")
