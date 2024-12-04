import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import griddata

# 设置参数
N = 2000000  # 总球数
#N =6.02E23*0.1*1/1000/150000/1000
Ms = 10000   # 绿球数
#Ms = 6.02E23*0.1*1/1000/150000/1000/10000
Ms_postive_ratio = 1 - np.exp(-Ms/N)
error_limit =0.2
# 生成n_values的对数范围值
n_values = np.logspace(np.log10(1), np.log10(N-1), num=500, dtype=int)

# 生成M/N的对数范围值，直接计算M/N的比例
postive_ratios = np.logspace(np.log10(1/N), np.log10(1-1/N), num=500)

results = []

for postive_ratio in postive_ratios:
    M = int(postive_ratio * N)  # 计算对应的M值
    for n in n_values:
        # 使用超几何分布公式的近似计算抽中红球和绿球的期望数量和标准差
        M_mean = postive_ratio * n
        M_std = np.sqrt(n * postive_ratio * (1 - postive_ratio) * (N - n) / (N - 1))
        
        Ms_mean = Ms_postive_ratio * n
        Ms_std = np.sqrt(n * Ms_postive_ratio * (1 - Ms_postive_ratio) * (N - n) / (N - 1))
        
        # 计算relative_error
        if Ms_mean == 0 or M_mean == 0:
            relative_error = np.nan
        else:
            relative_error = np.sqrt((M_std/M_mean)**2 + (Ms_std/Ms_mean)**2)

        # 仅保存relative_error小于0.2的结果
        if relative_error < error_limit:
            results.append((postive_ratio, n / N, relative_error))

# 将结果写入CSV文件
csv_filename = '6_5sampling_results_log_range_M_N_filtered2.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['postive_ratio', 'n / N', 'relative_error'])
    writer.writerows(results)

print(f'Results written to {csv_filename}')

# 提取用于绘图的数据
postive_ratios, n_N_values, relative_errors = zip(*results)

# 创建插值网格
grid_M_N, grid_n_N = np.meshgrid(np.unique(postive_ratios), np.unique(n_N_values))

# 使用griddata进行插值
grid_relative_errors = griddata((postive_ratios, n_N_values), relative_errors, (grid_M_N, grid_n_N), method='cubic')

# 绘制等高线图
plt.figure(figsize=(10, 8))
cp = plt.contourf(grid_M_N, grid_n_N, grid_relative_errors, levels=np.linspace(np.nanmin(grid_relative_errors), np.nanmax(grid_relative_errors), num=50), cmap='viridis')
plt.colorbar(cp)  # 显示颜色条

# contours = plt.contour(grid_M_N, grid_n_N, grid_relative_errors, levels=[0.05,0.1], colors='white')
# contour_lines = plt.contour(X, Y, Z, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=2)  # 特定等高线


contours = plt.contour(grid_M_N, grid_n_N, grid_relative_errors, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=2)  # 特定等高线




plt.clabel(contours, inline=True, fontsize=15, fmt='     %1.2f')

N=int(N)
Ms=int(Ms)
plt.title(f'Relative Error Contour Plot (CV)\nN={N},Ms={Ms}', fontsize=15)
plt.xlabel('postive %', fontsize=15)
plt.ylabel('n/N', fontsize=15)
plt.xscale('log')
# plt.yscale('log')


# 在图上添加垂直虚线
# plt.axvline(x=131/N, color='red', linestyle='--', label='131/N')
# plt.axvline(x=0.2071, color='red', linestyle='--', label='0.2071')
# plt.axvline(x=0.1017, color='yellow', linestyle='--', label='0.1017')

dpi_value = 600
# plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('二次取样后的误差条件情况注意区分后面的颗粒的情况2.png', dpi=dpi_value)  # 保存图像时指定dpi

plt.show()
