
import numpy as np
import csv

# 设置参数
N = 2000000  # 总球数
error_up_limit = 0.15  # 错误限制

n_values = np.logspace(np.log10(1), np.log10(N), num=100)
pos_Ms_values = np.logspace(np.log10(1), np.log10(N-1), num=100)      #阳性ms球的范围
postive_ratios = np.logspace(np.log10(1/N), np.log10(1-1/N), num=100)

results = []

for postive_ratio in postive_ratios:
    # M = int(postive_ratio * N)  # 计算对应的M值
    for n in n_values:
        for pos_Ms in pos_Ms_values:
            # 使用公式将 pos_Ms 转换为 Ms，用于最终的保存
            Ms = pos_Ms
            
            # 计算使用 pos_Ms 的 Ms_mean 和 Ms_std
            Ms_mean = pos_Ms / N * n
            Ms_std = np.sqrt(n * (pos_Ms / N) * (1 - pos_Ms / N) * (N - n) / (N - 1))
            
            # 计算 M 的均值和标准差
            M_mean = postive_ratio * n
            M_std = np.sqrt(n * postive_ratio * (1 - postive_ratio) * (N - n) / (N - 1))
            
            # 计算相对误差 relative_error
            if Ms_mean == 0 or M_mean == 0:
                relative_error = np.nan
            else:
                relative_error = np.sqrt((M_std/M_mean)**2 + (Ms_std/Ms_mean)**2)
            
            # 仅保存 relative_error 小于 0.5 的结果
            if relative_error < error_up_limit:
                results.append((Ms, postive_ratio, n / N, relative_error))

# 将结果写入 CSV 文件
csv_filename = '8_Ms_n_N_M_N_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ms', 'postive_ratio', 'n_N_ratio', 'relative_error'])
    writer.writerows(results)
