# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:35:12 2024

@author: lihua
"""

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Read data from CSV file
filename = '8_Ms_n_N_M_N_data.csv'
data = pd.read_csv(filename)

# Extract columns
Ms = data['Ms'].values
postive_ratio = data['postive_ratio'].values
n_N = data['n_N_ratio'].values
relative_error = data['relative_error'].values

# Convert to logarithmic scale
x = np.log10(postive_ratio)
y = np.log10(n_N)
z = np.log10(Ms)

# Create a grid
xi = np.linspace(np.min(x), np.max(x), 5)
yi = np.linspace(np.min(y), np.max(y), 5)
zi = np.linspace(np.min(z), np.max(z), 5)
xGrid, yGrid, zGrid = np.meshgrid(xi, yi, zi, indexing='ij')

# Interpolate the data
points = np.column_stack((x, y, z))
values = relative_error

# Linear interpolation
F_linear = LinearNDInterpolator(points, values)
vGrid_linear = F_linear(xGrid, yGrid, zGrid)

# Nearest neighbor interpolation for extrapolation
F_nearest = NearestNDInterpolator(points, values)
vGrid_nearest = F_nearest(xGrid, yGrid, zGrid)

# Combine linear interpolation with nearest neighbor extrapolation
vGrid = np.where(np.isnan(vGrid_linear), vGrid_nearest, vGrid_linear)

# Choose isosurface value
isosurface_value = 0.1  # Adjust this value based on your data and requirements

# Use marching cubes to find the surface mesh of the isosurface
# marching_cubes expects the data in (z, y, x) order
vGrid_t = np.transpose(vGrid, (2, 1, 0))

# Apply marching cubes
verts, faces, normals, values = marching_cubes(vGrid_t, isosurface_value, spacing=(zi[1]-zi[0], yi[1]-yi[0], xi[1]-xi[0]))

# Adjust the vertices to match the original grid
verts[:, 0] += xi[0]
verts[:, 1] += yi[0]
verts[:, 2] += zi[0]


# 3D Scatter plot with colors representing relative error
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=relative_error, cmap='viridis', alpha=1)
# plt.colorbar(sc, ax=ax, label='CV')
# cbar = plt.colorbar(sc, ax=ax, pad=0.01)  # 增大 pad 值将 colorbar 向右移动
# 调整色条的长度和与图形的间距
cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)

# 设置字体大小
label_fontsize = 15
title_fontsize = 20
cbar_label_fontsize = 15
tick_fontsize = 15  # 刻度标签字体大小

cbar.set_label('CV', fontsize=cbar_label_fontsize)
ax.set_xlabel('log10(postive %)', labelpad=5, fontsize=label_fontsize)
ax.set_ylabel('log10(n/N)', labelpad=5, fontsize=label_fontsize)
ax.set_zlabel('log10(Ms)', labelpad=1, fontsize=label_fontsize)
ax.set_title('3D Scatter Plot with Colors Representing CV', fontsize=title_fontsize)

# 调整颜色栏刻度标签的字体大小
cbar.ax.tick_params(labelsize=tick_fontsize)

# 调整刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.tick_params(axis='z', which='major', labelsize=tick_fontsize)

ax.view_init(elev=30, azim=-135)

# 调整布局
fig.tight_layout()
# plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.1)  # 调整边距

dpi_value = 300
# plt.savefig('3d_scatter_plot.png', dpi=dpi_value, bbox_inches='tight') 
plt.savefig('3d_scatter_plot.png', dpi=dpi_value)  # 保存图片并设置DPI
plt.show()









