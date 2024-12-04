# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:55:34 2024

@author: lihua
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:11:46 2024
@author: lihua
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
# 定义方程组
def equations(vars, q1, q2, K1, K2, K3, K4):
    x1, x2, x3, x4, x5, x6 = vars
    k1, k2, k3, k4 = K1, K2, K3, K4
    k5, k6, k7, k8 = K3, K4, K1, K2
    eq1 = k3*x2*x6 + k7*x3*x5 - (k4*x1 + k8*x1)
    eq2 = k1*x4*x5 + k4*x1 - (k2*x2 + k3*x2*x6)
    eq3 = k5*x4*x6 + k8*x1 - (k6*x3 + k7*x3*x5)
    eq4 = x1 + x2 + x3 + x4 - p
    eq5 = x1 + x2 + x5 - q1
    eq6 = x1 + x3 + x6 - q2
    return [eq1, eq2, eq3, eq4, eq5, eq6]

# 固定 q1 和 q2 的值
q1 = 1E-9  # mol/L
q2 = 1E-9  # mol/L
p = 1E-18  # mol/l

# 设置 K1 和 K2 的值的范围
K1_values = np.logspace(4, 6, 100)
K2_values = np.logspace(-5, -3, 100)
K3_values = np.logspace(4, 6, 3)
K4_values = np.logspace(-5, -3, 3)

# 初始化 x1 的值
x1_values = np.zeros((len(K1_values), len(K2_values), len(K3_values), len(K4_values)))

# 数值求解
for i, K1 in enumerate(K1_values):
    for j, K2 in enumerate(K2_values):
        for m, K3 in enumerate(K3_values):
            for n, K4 in enumerate(K4_values):
                # 估计初值
                initial_guess = [0, 0, 0, p, q1, q2]
                # 解方程
                solution = fsolve(equations, initial_guess, args=(q1, q2, K1, K2, K3, K4), xtol=1e-15, maxfev=100000)
                # 存储 x1 的值
                x1_values[i, j, m, n] = solution[0]

# 将 K1 和 K2 转换为对数刻度，以便绘图
K1_log = np.log10(K1_values)
K2_log = np.log10(K2_values)

# 为 K3 和 K4 准备标签
K3_labels = ['K3=%.1e' % K3 for K3 in K3_values]
K4_labels = ['K4=%.1e' % K4 for K4 in K4_values]


global_min = np.min(x1_values)
global_max = np.max(x1_values)
norm = Normalize(vmin=global_min, vmax=global_max)

# 创建子图网格
# fig, axes = plt.subplots(len(K3_values), len(K4_values), figsize=(15, 10), sharex=True, sharey=True)

# 定义 x1/p 的比率等高线
ratios = sorted([0.99,0.95,0.9,0.8,0.7,0.6, 0.5, 0.40, 0.30, 0.20, 0.10])
levels = [p * ratio for ratio in ratios]  # 使用 p 值来计算等高线值
fig, axes = plt.subplots(len(K3_values), len(K4_values), figsize=(18, 15), sharex=True, sharey=True)

# 遍历每个 K3 和 K4 的组合
for m, K3 in enumerate(K3_values):
    for n, K4 in enumerate(K4_values):
        ax = axes[m, n]
        # 创建 K1 和 K2 的网格
        X, Y = np.meshgrid(K1_log, K2_log)
        # 获取对应的 x1_values 切片，并转置以正确匹配坐标轴
        Z = x1_values[:, :, m, n].T
        # 绘制等高线图
        contour = ax.contourf(X, Y, Z, levels=100, cmap=cm.viridis, norm=norm)
        # 添加颜色条
        cbar = fig.colorbar(contour, ax=ax, norm=norm)
        # cbar = fig.colorbar(contour, ax=ax, cmap=cm.viridis, boundaries=np.linspace(global_min, global_max, 21))
        # 添加比率等高线
        CS = ax.contour(X, Y, Z, levels, colors='white')
        # 设置等高线标签
        fmt = {lvl: "{:.0f}%".format(ratio * 100) for lvl, ratio in zip(levels, ratios)}
        ax.clabel(CS, inline=True, fontsize=8, fmt=fmt)
        # 设置标题和标签
        ax.set_title(f'Kon2 (M^-1s^-1)={K3:.1e}, Koff2 (s^-1)={K4:.1e}')
        ax.set_xlabel('Kon1 (M^-1s^-1)')
        ax.set_ylabel('Koff1 (s^-1)')


# 调整布局并显示图形
dpi_value = 300
plt.tight_layout()
plt.savefig('图五动力学图ab与ag.png', dpi=dpi_value)

plt.show()
