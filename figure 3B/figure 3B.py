# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:54:16 2024

@author: lihuan
"""
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 读取数据
filename = '5.5_relative_error_between_lam_and_Poscent.csv'
data = pd.read_csv(filename)

# 提取各列数据
Ms = data['Ms']
M_N_ratio = data['M_N_ratio']
N = data['N']
relative_error = data['relative_error']

# 数据有效性检查
valid_idx = ~(Ms.isna() | M_N_ratio.isna() | N.isna() | relative_error.isna())
Ms = Ms[valid_idx]
M_N_ratio = M_N_ratio[valid_idx]
N = N[valid_idx]
relative_error = relative_error[valid_idx]

# 转换为对数坐标
x = np.log10(M_N_ratio)
y = np.log10(N)
z = np.log10(Ms)

# 构建插值器
points = np.vstack((x, y, z)).T
values = relative_error
interpolator = LinearNDInterpolator(points, values)

# 构建网格
xGrid, yGrid, zGrid = np.meshgrid(np.linspace(min(x), max(x), 5),
                                  np.linspace(min(y), max(y), 5),
                                  np.linspace(min(z), max(z), 5))

# 计算插值值
vGrid = interpolator(xGrid, yGrid, zGrid)

# 绘制散点图
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=relative_error, cmap='viridis', alpha=1)
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)  # 增大 pad 值将 colorbar 向右移动
cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
# 设置字体大小
label_fontsize = 15
title_fontsize = 20
cbar_label_fontsize = 15
tick_fontsize = 15  # 刻度标签字体大小

cbar.set_label('CV', fontsize=cbar_label_fontsize)
ax.set_xlabel('log10(M/N)', labelpad=5, fontsize=label_fontsize)
ax.set_ylabel('log10(N)', labelpad=5, fontsize=label_fontsize)
ax.set_zlabel('log10(Ms)', labelpad=1, fontsize=label_fontsize)
ax.set_title('3D Scatter Plot with Colors Representing CV', fontsize=title_fontsize)

# 调整颜色栏刻度标签的字体大小
cbar.ax.tick_params(labelsize=tick_fontsize)

# 调整刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.tick_params(axis='z', which='major', labelsize=tick_fontsize)

ax.view_init(elev=30, azim=135)

# 调整布局
fig.tight_layout()

dpi_value = 300
plt.savefig('3d_scatter_plot.png', dpi=dpi_value, bbox_inches='tight')  # 保存图片并设置DPI
plt.show()
