import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

# 定义最大值
Nmax = 500000

# 定义N和M的范围
N_values = np.logspace(np.log10(100000), np.log10(Nmax), num=50, dtype=int)
M_values = np.logspace(np.log10(1), np.log10(-np.log(1 / Nmax) * Nmax), num=50, dtype=int)

# 创建网格 N 和 M
N, M = np.meshgrid(N_values, M_values)

# 计算 K 的值
K = N * (1 - np.exp(-M / N))

# 查看K的最大值和最小值
print(f"K 最小值: {K.min()}, 最大值: {K.max()}")

# 绘制等高线图
plt.figure(figsize=(10, 8))

# 使用LogNorm来设置色条为对数尺度
contour = plt.contourf(N, M, K, levels=np.logspace(np.log10(K.min()), np.log10(K.max()), num=20), cmap='viridis', norm=LogNorm())

# 色条
cbar = plt.colorbar(contour)
cbar.set_label('K', fontsize=15)

# 设置对数刻度的范围
ticks = np.logspace(np.log10(K.min()), np.log10(K.max()), num=10)
cbar.set_ticks(ticks)  # 设置色条的刻度

# 强制使用科学计数法显示刻度
formatter = ScalarFormatter()
formatter.set_scientific(False)  # 始终使用科学计数法
formatter.set_powerlimits((-1, 1))  # 设置科学计数法显示范围

# 应用formatter到色条
cbar.ax.yaxis.set_major_formatter(formatter)


# 设置自定义的等高线级别，所有等高线的颜色都设为白色
levels = [1.2040, 0.1, 0.01, 131 / Nmax]

# 绘制所有等高线，颜色都设为白色
for level in levels:
    contour_line = plt.contour(N, M, M / N, levels=[level], colors='white', linewidths=1)
    plt.clabel(contour_line, inline=True, fontsize=12, fmt='%.6f', colors='white')
    # plt.clabel(contour_line, inline=True, fontsize=15, fmt='%.2f', colors='white', inline_spacing=50)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)





# 设置刻度字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
cbar.ax.tick_params(labelsize=15)
plt.xlabel('N (log)', fontsize=15)
plt.ylabel('M (log)', fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.title(f'Relationship between M, N and K\nInitial Nmax={Nmax}', fontsize=15)
# 显示图形
plt.show()
