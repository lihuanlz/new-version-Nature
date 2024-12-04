# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:34:14 2024

@author: lihua
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import griddata

import math

def calculate_max_balls_on_sphere(big_radius, small_radius):
    max_balls = (4 * math.pi * big_radius ** 2) / (3 * small_radius ** 2)
    return max_balls

# 测试
# big_radius = 3 / 2  # 大球的半径
# small_radius = 0.2 / 2  # 小球的半径
# max_balls = int(calculate_max_balls_on_sphere(big_radius, small_radius)*0.1)#设定只可以结合10%。
# print("最多可以放置的小球数量为:", max_balls)

# 设置参数
magnetic_beads = 500000  # 磁珠总数
N = (magnetic_beads)
#N =6.02E23*0.1*1/1000/150000/1000
Y = 10000   # 绿球数
#Y = 6.02E23*0.1*1/1000/150000/1000/10000
error_limit =0.2
# 生成n_values的对数范围值
n_values = np.logspace(np.log10(1), np.log10(N-1), num=200)

# 生成M/N的对数范围值，直接计算M/N的比例
postive_ratios = np.logspace(np.log10(1/N), np.log10(1-1/N), num=200)

results = []

for postive_ratio in postive_ratios:
    M = int(postive_ratio * N)  # 计算对应的M值
    for n in n_values:
        # 使用超几何分布公式的近似计算抽中红球和绿球的期望数量和标准差
        M_mean = postive_ratio * n
        M_std = np.sqrt(n * postive_ratio * (1 - postive_ratio) * (N - n) / (N - 1))
        
        Y_mean = Y/N * n
        Y_std = np.sqrt(n * Y/N * (1 - Y/N) * (N - n) / (N - 1))
        
        # 计算relative_error
        if Y_mean == 0 or M_mean == 0:
            relative_error = np.nan
        else:
                    
            relative_error = np.sqrt((M_std/M_mean)**2 + (Y_std/Y_mean)**2)         #注意这里的参数是包含了解离过程的。
            
     # 仅保存relative_error小于0.2的结果
        if relative_error < error_limit:
            results.append((postive_ratio, n / N, relative_error))

# 将结果写入CSV文件
# 提取用于绘图的数据
postive_ratios, n_N_values, relative_errors = zip(*results)

# 将x轴的值替换为λ = -ln(1 - postive_ratios)
postive_ratios = -np.log(1 - np.array(postive_ratios))#################################
lambda_values = postive_ratios
print (postive_ratios)


# 将结果写入CSV文件
csv_filename = '6_5sampling_results_log_range_M_N_filtered2.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['postive_ratio', 'n / N', 'postive_ratios', 'relative_error'])  # Add 'postive_ratios' to the header
    # Write each row along with postive_ratios
    for result, lambda_value in zip(results, postive_ratios):
        writer.writerow([result[0], result[1], lambda_value, result[2]])

print(f'Results written to {csv_filename}')



# 创建插值网格
grid_M_N, grid_n_N = np.meshgrid(np.unique(postive_ratios), np.unique(n_N_values))

# 使用griddata进行插值
grid_relative_errors = griddata((postive_ratios, n_N_values), relative_errors, (grid_M_N, grid_n_N), method='cubic')

# 绘制等高线图
plt.figure(figsize=(10, 8))
cp = plt.contourf(grid_M_N, grid_n_N, grid_relative_errors, levels=np.linspace(np.nanmin(grid_relative_errors), np.nanmax(grid_relative_errors), num=50), cmap='viridis')
plt.colorbar(cp)  # 显示颜色条

contours = plt.contour(grid_M_N, grid_n_N, grid_relative_errors, levels=[0.05,0.1], colors='white')
plt.clabel(contours, inline=True, fontsize=15, fmt='     %1.2f')

N=int(N)
Y=int(Y)
plt.title(f'Relative Error Contour Plot (CV)\nN={N},Ms={Y}', fontsize=15)
plt.xlabel('λ', fontsize=15)
plt.ylabel('n/N', fontsize=15)
plt.xscale('log')
# plt.yscale('log')


# 在图上添加垂直虚线
plt.axvline(x=131/N, color='red', linestyle='--', label='131/N')
plt.axvline(x=1.2404, color='red', linestyle='--', label='0.2146')
# plt.axvline(x=0.1035, color='yellow', linestyle='--', label='0.1035')

dpi_value = 600
# plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('二次取样后的误差条件情况注意区分后面的颗粒的情况2.png', dpi=dpi_value)  # 保存图像时指定dpi

plt.show()
