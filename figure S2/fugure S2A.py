# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:12:48 2024

@author: lihua
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize

# 初始浓度和平衡常数
B0 = 1e-18  # 初始浓度 [B]_0

K1 = 1/(1e-9)# 平衡常数 K1
K2 = 1/(1e-9)  # 平衡常数 K2
K3 = (5.1e6)/(5.1e-9)  # 平衡常数 K3

# 定义一个函数来求解二次方程，同时选择合理的解
def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solution exists")
    
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    
    # 选择合理的解
    return max(0, min(root1, root2))

# 生成等高线图的数据
A0_range = np.logspace(-12, -7, 100)
C0_range = np.logspace(-12, -7, 100)
D0_values = [1e-9,1e-12, 1e-15, 1e-16]

for D0 in D0_values:
    ABCD_concentration = np.zeros((len(A0_range), len(C0_range)))
    
    for i, A0 in enumerate(A0_range):
        for j, C0 in enumerate(C0_range):
            # 第一步反应：A + B = AB
            a1 = K1
            b1 = -(K1 * (A0 + B0) + 1)
            c1 = K1 * A0 * B0

            x1 = solve_quadratic(a1, b1, c1)

            # 第二步反应：AB + C = ABC
            a2 = K2
            b2 = -(K2 * (x1 + C0) + 1)
            c2 = K2 * x1 * C0

            y1 = solve_quadratic(a2, b2, c2)

            # 第三步反应：ABC + D = ABCD
            a3 = K3
            b3 = -(K3 * (y1 + D0) + 1)
            c3 = K3 * y1 * D0

            z1 = solve_quadratic(a3, b3, c3)

            # 存储 [ABCD] 的浓度
            ABCD_concentration[i, j] = z1

    # 绘制等高线图
    A0_grid, C0_grid = np.meshgrid(A0_range, C0_range)
    plt.figure()
    levels = np.linspace(0, B0, 50)  # 设置等高线的级别
    norm = Normalize(vmin=0, vmax=B0)  # 设置颜色归一化
    cp = plt.contourf(A0_grid, C0_grid, ABCD_concentration.T, levels=levels, cmap='viridis', norm=norm)
    plt.colorbar(cp)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ab1 (M)')
    plt.ylabel('Ab2 (M)')
    plt.title(f'Complex concentration for SPA = {D0} M')

    # 绘制特定等高线 [ABCD]/B0 = 0.9, 0.8, 0.7, 0.6
    specific_levels = [0.9 * B0, 0.8 * B0, 0.7 * B0, 0.6 * B0, 0.5 * B0, 0.4 * B0, 0.3 * B0, 0.2 * B0, 0.1 * B0, 0.05 * B0, 0.01 * B0]
    specific_levels.sort()  # 确保 levels 是递增的
    cs = plt.contour(A0_grid, C0_grid, ABCD_concentration.T, levels=specific_levels, colors='white', linestyles='dashed')
    plt.clabel(cs, fmt={level: f'{(level/B0)*100:.1f}%' for level in specific_levels}, inline=True, fontsize=8)


    # Save the plot as PNG file with 300 dpi
    filename = f'contour_plot_D0_{D0}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()  # Close the figure after saving to avoid memory issues
