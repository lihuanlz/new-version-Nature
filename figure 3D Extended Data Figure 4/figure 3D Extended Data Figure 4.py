

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math
from scipy.special import i0
# 定义参数
N = 500000
Ms = 10000
# a_values = 0



error_limit =1
# 生成M/N的对数范围值，直接计算M/N的比例
M_N_ratios = np.logspace(np.log10(1/N), np.log10(-math.log(1/N)), num=100)
# 生成n_values的对数范围值
n_values = np.logspace(np.log10(1), np.log10(N), num=100)
# 初始化满足条件的最大i值
max_postive_beads = 0
Ms_postive_ratio = 1 - np.exp(-Ms/N)

# 用于存储计算结果的列表
results = []

# 遍历所有可能的postive_beads值
for M_N_ratio in M_N_ratios:  # 直接迭代M_N_ratios数组
    C1 = (1-np.exp(-M_N_ratio)) / (1-np.exp(-Ms/N))
    # C1 = (1 - np.exp(-M_N_ratio * (1 + a_values)) * i0(2 * M_N_ratio * np.sqrt(a_values))) / (1 - np.exp(-Ms/N * (1 + a_values)) * i0(2 * Ms/N * np.sqrt(a_values)))

    C2 = M_N_ratio * N / Ms
    error = np.abs((C1 - C2) / C2)
    
    # 将结果添加到列表中
    results.append([M_N_ratio, error])

    if error < error_limit:
        max_postive_beads = M_N_ratio * N
    else:
        break

# 写入CSV文件
with open('7_error.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['M/N ratio', 'error'])  # 写入头部
    for result in results:
        writer.writerow(result)

print('CSV file has been written with the 7_error.csv.')



results = []

for M_N_ratio in M_N_ratios:
    postive_ratio = 1 - np.exp(-M_N_ratio)
    for n in n_values:
        # 计算抽中红球和绿球的期望数量和标准差
        M_mean = postive_ratio * n #M_mean是阳性的数目
        M_std = np.sqrt(n * postive_ratio * (1 - postive_ratio) * (N - n) / (N - 1))
        
        #Ms_mean = (1 - postive_ratio) * n#注意这里有问题
        
        #Ms_std = np.sqrt(n * (1 - postive_ratio) * postive_ratio * (N - n) / (N - 1))
        Ms_mean = (1 - np.exp(-Ms/N)) * n
        Ms_std = np.sqrt(n * Ms_postive_ratio * (1-Ms_postive_ratio) * (N - n) / (N - 1))
        
        RSD = np.sqrt((M_std/M_mean)**2 + (Ms_std/Ms_mean)**2)
        
        if RSD < error_limit:
            results.append((M_N_ratio, n / N, RSD))

# 将结果写入CSV文件
csv_filename = '7_RSD.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['M_N_ratio', 'n / N', 'RSD'])
    writer.writerows(results)

print(f'Results written to {csv_filename}')



# 读取 7_error.csv 文件
error ={}
with open('7_error.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
        error[row[0]] = row[1]

# 读取并更新 7_RSD.csv 文件
with open('7_RSD.csv', 'r') as f_in, \
     open('7_updated_sampling_results1.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # 写入标题行
    header = next(reader)
    header.append('error')
    writer.writerow(header)

    # 更新数据行
    for row in reader:
        if row[0] in error:
            row.append(error[row[0]])
        else:
            row.append('N/A')
        writer.writerow(row)

print("File '7_updated_sampling_results.csv' has been created.")




# 读取 7_error.csv 文件
error ={}
with open('7_error.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
        error[row[0]] = row[1]

# 读取并更新 7_RSD.csv 文件
with open('7_RSD.csv', 'r') as f_in, \
     open('7_updated_sampling_results.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # 写入标题行
    header = next(reader)
    header.append('error')
    header.append('sum_of_last_two_columns')
    writer.writerow(header)

    # 更新数据行
    for row in reader:
        if row[0] in error:
            row.append(error[row[0]])
            #sum_of_last_two = float(row[-2]) + float(row[-1])
            
            
            
            
            sum_of_last_two = math.sqrt(float(row[-2])**2 + float(row[-1])**2) 
            
            
            
            
            
            #这里是最终的合成公式注意这里添加了合并解离的过程的误差
            row.append(sum_of_last_two)
        else:
            row.append('N/A')
            row.append('N/A')
        writer.writerow(row)

print("File '7_updated_sampling_results.csv' has been created.")

# 读取 7_updated_sampling_results.csv 文件
data = []
with open('7_updated_sampling_results.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
     try:
        x_value = float(row[0])
        y_value = float(row[1])
        z_value = float(row[4]) if row[4] != 'N/A' else np.nan  # 将'N/A'替换为np.nan
        data.append([x_value, y_value, z_value])
     except ValueError:
        # 这里可以打印错误信息或者忽略
        print(f"Skipping row due to conversion error: {row}")

# 将数据转换为NumPy数组
data = np.array(data)

# 提取x, y, z轴的数据
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# 创建网格
X, Y = np.meshgrid(np.unique(x), np.unique(y))

# 插值计算Z值
Z = griddata((x, y), z, (X, Y), method='linear')



# 绘制等高线图（填充）
plt.figure(figsize=(10, 8))
CSF = plt.contourf(X, Y, Z, 200, cmap='viridis')  # 填充等高线图
plt.colorbar(CSF)

# 在同一图上绘制特定的等高线（z=0.1）
CS = plt.contour(X, Y, Z, levels=[0.05,0.1], colors='white', linewidths=2, fontsize=15)  # 绘制特定等高线
plt.clabel(CS, inline=True, fontsize=15, fmt='%1.2f')  # 标注等高线的值

plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('n/N ratio', fontsize=15)
plt.title(f'Contour Plot with Error Limit\nN={N}, Ms={Ms}', fontsize=15)
plt.xscale('log')
#plt.yscale('log')

# 在图上添加垂直虚线
plt.axvline(x=131/N, color='gray', linestyle='--', label='131/N')
plt.axvline(x=1.2040, color='black', linestyle='--', label='1.2040')
plt.axvline(x=0.2071, color='yellow', linestyle='--', label='0.2071')

# 添加图例
dpi_value = 600
plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('图十三总误差图.png', dpi=dpi_value)  # 保存图像时指定dpi


plt.show()
