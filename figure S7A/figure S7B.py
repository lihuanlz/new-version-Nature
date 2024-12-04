import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
N = 2000000
Ms = 100000
error_limit = 0.1


# 计算Ms和N的值
N_values = np.logspace(np.log10(1), np.log10(N), num=250,dtype=int)
M_N_ratios = np.logspace(np.log10(1/N), np.log10(int(-np.log(1/N))), num=250)

# 准备数据列表，用于DataFrame
data_list = []

# 外层循环遍历不同的N值
for N in N_values:
    # 中层循环遍历不同的M_N_ratio值
    for M_N_ratio in M_N_ratios:
        M = M_N_ratio * N  # 计算M
        if M > 0 and (Ms/N) > 0:  # 确保分母不为零
            C1 = (1 - np.exp(-M / N)) / (1 - np.exp(-Ms / N))
            C2 = M / Ms
            relative_error = np.abs((C1 - C2) / C2)
            # 如果相对误差小于限定误差，则保存结果
            if relative_error < error_limit:
                data_list.append({'N': N, 'M_N_ratio': M_N_ratio,  'relative_error': relative_error})

# 将列表转换为DataFrame
df = pd.DataFrame(data_list)

# 使用pivot_table来创建用于绘图的网格数据
grid_df = df.pivot_table(index='N', columns='M_N_ratio', values='relative_error', aggfunc='min')

X, Y = np.meshgrid(grid_df.columns, grid_df.index)
Z = grid_df.values

plt.figure(figsize=(10, 8))
contour_filled = plt.contourf(X, Y, Z, levels=100, cmap='viridis', fontsize=15)
plt.colorbar(contour_filled)

contour_lines = plt.contour(X, Y, Z, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=2)  # 特定等高线
# plt.clabel(contour_lines, inline=False, fontsize=15, fmt='%1.2f')  # 标注等高线的值



plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('N', fontsize=15)
plt.title(f'Relative Error Contour Plot\nError Limit = {error_limit},Ms={Ms}', fontsize=15)
plt.xscale('log')
#plt.yscale('log')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# plt.axvline(x=131/N, color='gray', linestyle='--', label='131/N')
# plt.axvline(x=0.2071, color='black', linestyle='--', label='0.2071')
# plt.axvline(x=1.2040, color='yellow', linestyle='--', label='1.2040')
# plt.legend()

plt.savefig('图十一relative_error_contour_plot.png', dpi=600)
plt.show()

# 将结果DataFrame保存为CSV文件
df.to_csv('5_6relative_error_data.csv', index=False)
