import pandas as pd
import matplotlib.pyplot as plt
###
'''
可视化思路：直观比较强度用Mean作为纵坐标
同一个位置（如 DE 或 FE）的数据会包含多种故障类型（如 OR、IR、B），以及 正常数据。

传感器数据（DE、FE）进行比较，且使用相同的 采样频率（48kHz）。

对比每个 故障类型（OR、IR、B）与正常数据，确保 正常数据 和 故障数据 是在相同条件下（传感器类型和采样频率）进行对比。

固定坐标范围：

横坐标（slice_start）：基于切片的时间点。

纵坐标（mean）：设置了一个固定的纵坐标范围 ax1.set_ylim([-0.025, 0.175])，以确保无论信号的幅值大小如何变化，所有图表的纵坐标范围一致，便于比较。
'''
###
# 加载数据
df_A = pd.read_csv('task1_features_A_with_peaks.csv')

# 选择 DE 传感器的数据，正常信号和各故障信号（IR, OR, B）
normal_data_A_DE = df_A[(df_A['label_str'] == 'N') & (df_A['channel'] == 'DE')]
fault_data_A_IR_DE = df_A[(df_A['label_str'] == 'IR') & (df_A['channel'] == 'DE')]  # 内圈故障
fault_data_A_OR_DE = df_A[(df_A['label_str'] == 'OR') & (df_A['channel'] == 'DE')]  # 外圈故障
fault_data_A_B_DE = df_A[(df_A['label_str'] == 'B') & (df_A['channel'] == 'DE')]   # 滚动体故障

# 获取每个数据的切片起始与结束时间（假设故障数据和正常数据的时间范围相似）
x_range_start = 0
x_range_end = 200000  # 可以根据需要调整这个范围

# 可视化波形（选择某个切片进行绘制）
plt.figure(figsize=(14, 10))

# A 路线 DE 传感器的波形（左边是不同故障数据，右边是正常数据）
ax1 = plt.subplot(3, 2, 1)
ax1.plot(fault_data_A_IR_DE['slice_start'], fault_data_A_IR_DE['mean'], label='IR (Inner Race)', color='red')
ax1.set_xlim([x_range_start, x_range_end])  # 限制横坐标范围
ax1.set_ylim([-0.05, 0.175])  # 设置纵坐标固定范围
ax1.set_title('IR Signal (DE)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()

ax2 = plt.subplot(3, 2, 2)
ax2.plot(fault_data_A_OR_DE['slice_start'], fault_data_A_OR_DE['mean'], label='OR (Outer Race)', color='orange')
ax2.set_xlim([x_range_start, x_range_end])  # 限制横坐标范围
ax2.set_ylim([-0.05, 0.175])  # 设置纵坐标固定范围
ax2.set_title('OR Signal (DE)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.legend()

ax3 = plt.subplot(3, 2, 3)
ax3.plot(fault_data_A_B_DE['slice_start'], fault_data_A_B_DE['mean'], label='B (Ball)', color='green')
ax3.set_xlim([x_range_start, x_range_end])  # 限制横坐标范围
ax3.set_ylim([-0.05, 0.175])  # 设置纵坐标固定范围
ax3.set_title('B Signal (DE)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.legend()

ax4 = plt.subplot(3, 2, 4)
ax4.plot(normal_data_A_DE['slice_start'], normal_data_A_DE['mean'], label='Normal', color='blue')
ax4.set_xlim([x_range_start, x_range_end])  # 限制横坐标范围
ax4.set_ylim([-0.05, 0.175])  # 设置纵坐标固定范围
ax4.set_title('Normal Signal (DE)')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude')
ax4.legend()

# 自动调整图像布局
plt.tight_layout()
plt.show()
