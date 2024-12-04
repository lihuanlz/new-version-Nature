# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:58:36 2024

@author: lihua
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

# 读取CSV文件
df = pd.read_csv('confidence_intervals_with_ratios.csv')

# 设置显著性水平 alpha
alpha = 0.05        # 显著性水平，95% 置信区间

# 用于存储计算结果的列表
final_lambda_lower_list = []
final_lambda_upper_list = []

# 遍历每一行数据，计算 lambda 的置信区间
for index, row in df.iterrows():
    # 从数据行中获取样本大小 n
    n = row['N']  # 假设 CSV 文件中的第一列是 'Sample Size'
    
    lower_bound_lambda = row['Lower Bound lamuda']
    upper_bound_lambda = row['Upper Bound lamuda']

    # 计算对应的 observed_events
    lower_bound_events = lower_bound_lambda * n
    upper_bound_events = upper_bound_lambda * n

    # 使用 Poisson 分布的反向累积分布函数计算 Lower Bound λ 的 95% 置信区间
    lower_bound_lambda_lower = poisson.ppf(alpha / 2, lower_bound_events) / n
    lower_bound_lambda_upper = poisson.ppf(1 - alpha / 2, lower_bound_events) / n

    # 使用 Poisson 分布的反向累积分布函数计算 Upper Bound λ 的 95% 置信区间
    upper_bound_lambda_lower = poisson.ppf(alpha / 2, upper_bound_events) / n
    upper_bound_lambda_upper = poisson.ppf(1 - alpha / 2, upper_bound_events) / n

    # 最终的置信区间：Lower Bound λ 的下限与 Upper Bound λ 的上限
    final_lambda_lower = lower_bound_lambda_lower
    final_lambda_upper = upper_bound_lambda_upper

    final_lambda_lower_list.append(final_lambda_lower)
    final_lambda_upper_list.append(final_lambda_upper)

# 将计算结果添加到DataFrame中
df['final_lambda_lower'] = final_lambda_lower_list
df['final_lambda_upper'] = final_lambda_upper_list

# 将结果保存为新的CSV文件
df.to_csv('updated_confidence_intervals.csv', index=False)

# 显示处理后的数据
print(df.head())
