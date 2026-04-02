import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# M/D/1 Parameters
# -----------------------------
service_time = 0.045   # 45 ms
mu = 1 / service_time  # service rate (per second)

# Inter-arrival times (100ms to 50ms)
inter_arrival_times = np.arange(0.1, 0.049, -0.005)  # seconds
lam = 1 / inter_arrival_times  # arrival rate
rho = lam / mu

# -----------------------------
# M/D/1 formulas
# -----------------------------
N = (rho / (1 - rho)) * (1 - rho / 2)
Nq = N - rho

T = (1 / mu / (1 - rho)) * (1 - rho / 2)
W = T - (1 / mu)
W_ms = W * 1000

# -----------------------------
# 75%负载对应点
# -----------------------------
rho_target = 0.75
lam_75 = rho_target * mu
# 对应队列长度和等待时间（插值）
Nq_75 = np.interp(lam_75, lam, Nq)
W_75_ms = np.interp(lam_75, lam, W_ms)

# -----------------------------
# Plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴: Queue length
ax1.plot(lam, Nq, marker='o', color='blue',
         label="Average Number in Queue (Nq)")
ax1.set_xlabel("Arrival Rate λ (requests per second)",
               fontsize=14, fontweight='bold')
ax1.set_ylabel("Average Number in Queue (Nq)",
               fontsize=14, fontweight='bold', color='blue')
ax1.set_ylim(bottom=0)

# 右轴: Waiting time
ax2 = ax1.twinx()
ax2.plot(lam, W_ms, marker='s', linestyle='--', color='red',
         label="Average Waiting Time (ms)")
ax2.set_ylabel("Average Waiting Time (ms)",
               fontsize=14, fontweight='bold', color='red')
ax2.set_ylim(bottom=0)

# -----------------------------
# 75%负载对应点
# -----------------------------
ax1.plot(lam_75, Nq_75, 'o', color='blue')
ax2.plot(lam_75, W_75_ms, 'o', color='red')

# 标注
ax1.text(lam_75 + 0.2, Nq_75, f"Nq={Nq_75:.2f}", color='blue')
ax2.text(lam_75 + 0.2, W_75_ms, f"W={W_75_ms:.0f} ms", color='red')

# -----------------------------
# Legend
# -----------------------------
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.title("Queue Performance vs Arrival Rate",
          fontsize=16, fontweight='bold')

plt.grid(True)
plt.tight_layout()
plt.savefig("queue_performance.png", dpi=300, bbox_inches='tight')
plt.show()