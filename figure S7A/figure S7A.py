import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
N = 2000000

# 定义Ms的不同取值
Ms_values = np.logspace(np.log10(1), np.log10(N*np.exp(1/N)), num=100)

# 生成M/N的对数范围值
M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=100)

# 用于存储所有计算结果的字典
results ={}# 用于存储每个Ms下的最小expression_value及其对应的M/N ratio
min_values ={}# 外层循环遍历不同的Ms值
for Ms in Ms_values:
    results[Ms] = []
    min_val = float('inf')  # 初始化最小值为无穷大
    min_ratio = None  # 初始化对应的M/N ratio
    for M_N_ratio in M_N_ratios:
        C1 = (1 - np.exp(-M_N_ratio)) / (1 - np.exp(-Ms/N))
        C2 = M_N_ratio * N / Ms
        expression_value = np.abs((C1 - C2) / C2)
        results[Ms].append(expression_value)
        if expression_value < min_val:
            min_val = expression_value
            min_ratio = M_N_ratio
    min_values[Ms] = (min_val, min_ratio)

# 准备数据用于绘图
Ms_array = np.array(Ms_values)
M_N_ratios_array = np.array(M_N_ratios)
expression_values_array = np.array([results[Ms] for Ms in Ms_values])

# 绘制等高线图
X, Y = np.meshgrid(M_N_ratios_array, Ms_array)
Z = expression_values_array

plt.figure(figsize=(10, 8))
contour_filled = plt.contourf(X, Y, Z, levels=100, cmap='viridis')  # 背景填充等高线
plt.colorbar(contour_filled)
contour_lines = plt.contour(X, Y, Z, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=2, fontsize=15)  # 特定等高线
# plt.clabel(contour_lines, inline=False, fontsize=10, fmt='%1.2f')  # 标注等高线的值
plt.xscale('log')
plt.yscale('log')
plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('Ms', fontsize=15)
plt.title(f'Expression Value Contour Plot with Error\nN={N}', fontsize=15)

# 标记每个Ms条件下最小expression_value的值
# 准备数据列表
min_ratios = []
Ms_values = []

# 遍历字典，填充数据列表
for Ms, (min_val, min_ratio) in min_values.items():
    min_ratios.append(min_ratio)
    Ms_values.append(Ms)

# 绘制线条和点
plt.plot(min_ratios, Ms_values, 'k-')  # 这将连接所有点



# 在图上添加垂直虚线
# plt.axvline(x=131/N, color='gray', linestyle='--', label='131/N')
# plt.axvline(x=0.2071, color='black', linestyle='--', label='0.2071')
# plt.axvline(x=1.2040, color='yellow', linestyle='--', label='1.2040')

dpi_value = 600
# plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('图十一计算所有偏差的3d图找最佳的条件.png', dpi=dpi_value)  # 保存图像时指定dpi

# 显示图像
plt.show()



# 创建一个空的列表，用于存储数据
csv_data = []

# 遍历 min_values 字典，并将 Ms, min_ratio 和 min_val 保存下来
for Ms, (min_val, min_ratio) in min_values.items():
    csv_data.append([Ms, min_ratio, min_val])

# 将列表转换为 pandas DataFrame
df = pd.DataFrame(csv_data, columns=['Ms', 'min_ratio', 'expression_value'])

# 将 DataFrame 保存为 CSV 文件，不保存行索引
df.to_csv('5expression_values2.csv', index=False)


