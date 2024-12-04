# -*- coding: utf-8 -*- 
"""Created on Thu Nov 28 13:39:27 2024@author: lihua"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 导入 LogNorm
from matplotlib.ticker import ScalarFormatter
# 定义 K 和 N 的范围
Nmax = 2000000
N_values = np.logspace(1, np.log10(Nmax), num=50)
K_values = np.logspace(1, np.log10(Nmax), num=100)
# K_values = np.linspace(1, Nmax, num=Nmax-1)  # K 采用线性分布
# 创建网格 K, N
K, N = np.meshgrid(K_values, N_values)

# 计算 M 值
M = -N * np.log(1 - K / N)

# 绘制等高线图
plt.figure(figsize=(10, 8))

# 增加 levels 参数的数量，例如设置为 200，来增加颜色的细腻度
# cp = plt.contourf(K, N, M, levels=200, cmap='viridis', norm=LogNorm())  # 设置 norm 为 LogNorm
contour = plt.contourf(N, K, M, levels=np.logspace(np.log10(K.min()), np.log10(K.max()), num=10), cmap='viridis', norm=LogNorm())
# 添加颜色条，显示为对数刻度
cbar = plt.colorbar(contour)
cbar.set_label('M', fontsize=15)

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
levels = [1.204, 0.1, 0.01, 131 / Nmax]

# 绘制所有等高线，颜色都设为白色
for level in levels:
    contour_line = plt.contour(N, M, M / N, levels=[level], colors='white', linewidths=1)
    plt.clabel(contour_line, inline=True, fontsize=12, fmt='%.6f', colors='white')
    # plt.clabel(contour_line, inline=True, fontsize=15, fmt='%.2f', colors='white', inline_spacing=50)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)










plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# 图标题及标签

plt.xscale('log')
plt.yscale('log')
plt.xlabel('N', fontsize=15)
plt.ylabel('K', fontsize=15)
plt.title(f'Relationship between M, N and K\nInitial Nmax={Nmax}', fontsize=15)
# 显示图像
plt.show()
