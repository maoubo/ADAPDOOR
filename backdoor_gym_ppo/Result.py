import matplotlib.pyplot as plt
import numpy as np

"""
settings: learn from scratch + action poisoning + reward hacking
更改reward的效果
"""
# Normal = [486.13, 488.55, 490.95, 490.9, 376, 217.43, 150.96, 54.07]
# ASR = [0, 0.4936, 0.9215, 0.9888, 0.996, 0.9968, 0.9984, 0.9912]
#
# labels = ["0.5", "1", "5", "10", "50", "100", "1000", "10000"]
# fig, ax = plt.subplots()
# ax_ = ax.twinx()  # 右边y轴
# ax.set_title("learn from scratch + action poisoning + reward hacking", fontsize=16)
# ax.set_xlabel('Backdoor Reward', fontsize=16)
# ax.set_ylabel('Normal Performance', fontsize=16)
# ax_.set_ylabel('ASR', fontsize=16)
#
# ax.plot(labels, Normal, '-.', color="gray", markersize=10, linewidth=4, label="Normal Performance")
# ax_.plot(labels, ASR, 's--', color=[0.98, 0.84, 0.66], markersize=10, linewidth=4, label="ASR")
#
# fig.legend(loc="center")
# plt.show()

"""
settings: learn from scratch + action poisoning + reward hacking
更改训练步长的效果
"""
# reward = 100
# Normal = [217.4309, 221.8644, 278.2183, 426.3954, 480.8514, 488.9219]
# ASR = [0.9968, 0.9992, 1.0, 0.9984, 0.9984, 0.9984]
#
# # reward = 5
# # Normal = [493.8387, 490.254, 472.8919, 500.0, 500.0, 481.4179]
# # ASR = [0.9151, 0.9856, 0.9415, 0.9864, 0.6218, 0.9808]
#
# # reward = 2
# # Normal = [484.353, 488.0834, 488.9714, 471.36, 489.9842, 497.9166]
# # ASR = [0.5369, 0.5657, 0.5665, 0.4904, 0.5529, 0.5296]
#
# # reward = 10000
# # Normal = [53.2788, 48.5219, 51.2149, 59.6459, 59.055, 60.3291]
# # ASR = [0.988, 0.9936, 0.9936, 0.9936, 0.996, 0.996]
#
# labels = [500000, 600000, 700000, 800000, 900000, 1000000]
# fig, ax = plt.subplots()
# ax_ = ax.twinx()  # 右边y轴
# ax.set_title("backdoor-reward = 2", fontsize=16)
# ax.set_xlabel('Steps', fontsize=16)
# ax.set_ylabel('Normal Performance', fontsize=16)
# ax_.set_ylabel('ASR', fontsize=16)
# ax.set_ylim(0, 500)
# ax_.set_ylim(0, 1)
#
# ax.plot(labels, Normal, '-.', color="gray", markersize=10, linewidth=4, label="Normal Performance")
# ax_.plot(labels, ASR, 's--', color=[0.98, 0.84, 0.66], markersize=10, linewidth=4, label="ASR")
#
# fig.legend(loc="center")
# plt.show()

"""
settings: reward hacking
更改训练步长的效果
"""
# learn from scratch
# Normal = [318.9099, 223.7159, 264.4189, 303.8385, 475.8718, 480.8816]
# ASR = [1.0, 0.9992, 1.0, 1.0, 1.0, 0.9984]

# trained model
# Normal = [171.8502, 425.4432, 478.7102, 385.9118, 479.9359, 448.2706]
# ASR = [0.9792, 0.9976, 1.0, 0.9984, 1.0, 0.9984]
#
# labels = [500000, 600000, 700000, 800000, 900000, 1000000]
# fig, ax = plt.subplots()
# ax_ = ax.twinx()  # 右边y轴
# ax.set_title("trained model", fontsize=16)
# ax.set_xlabel('Steps', fontsize=16)
# ax.set_ylabel('Normal Performance', fontsize=16)
# ax_.set_ylabel('ASR', fontsize=16)
# ax.set_ylim(0, 500)
# ax_.set_ylim(0, 1)
#
# ax.plot(labels, Normal, '-.', color="gray", markersize=10, linewidth=4, label="Normal Performance")
# ax_.plot(labels, ASR, 's--', color=[0.98, 0.84, 0.66], markersize=10, linewidth=4, label="ASR")
#
# fig.legend(loc="center")
# plt.show()

"""
更改reward的效果
"""
Normal = [1.0, 0.6513, 0.4118, 0.7018, 0.7072, 0.2761, 0.2375]
ASR = []
ASR.append([[0.879], [0.9623], [0.3982], [0.9679], [0.9804], [1.0], [1.0]])
ASR.append([[0.9363], [0.9179], [0.9996], [0.9896], [0.9872], [1.0], [1.0]])
ASR.append([[0.9355], [1.0], [0.9728], [1.0], [1.0], [1.0], [1.0]])

labels = ["1", "5", "10", "50", "100", "1000", "10000"]
fig, ax = plt.subplots()
ax.set_title("Learn from Scratch", fontsize=16)
ax.set_xlabel('Backdoor Reward', fontsize=16)
ax.set_ylabel('Normal Performance', fontsize=16)
ax.set_ylim(0, 1.05)

ax.plot(labels, Normal, '-.', color="gray", markersize=10, linewidth=4, label="Normal Performance")
for i in range(len(ASR)):
    ax.plot(labels, ASR[i], markersize=10, linewidth=4, label="ASR {}".format(i))

fig.legend(loc="center")
plt.show()