# -*- coding: utf-8 -*-  
"""Created on Sun Nov 17 17:18:36 2024
@author: lihua"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 总体大小
N = 2000000
λ1 = 131 / N
# λ2 = 0.2071
λ2 = 1.2040

# 计算λ对应的x值
x1 = (1 - np.exp(-λ1))*100
# x2 = (1 - np.exp(-λ2))*100
x3 = (1 - np.exp(-λ2))*100

# 初始CV = 0.1
CV_1 = 0.1
# 新增 CV = 0.05
CV_2 = 0.05
# 使用对数分布生成 ratios
ratios = np.logspace(np.log10(1 / N), np.log10((N - 1) / N), 20)

# 计算第一个CV值对应的最小n和kn_ratio_product
def calculate_min_n_and_kn_ratio_product(CV, ratios, N):
    results = {}
    for ratio in ratios:
        K = int(N * ratio)  # 根据比值计算K
        # 寻找满足条件的最小n值
        for n in range(1, N + 1):
            left_side = CV * n * (K / N)  # 均值与CV的乘积
            right_side = np.sqrt(n * (K / N) * (1 - K / N) * ((N - n) / (N - 1)))  # 标准差
            if left_side > right_side:
                results[ratio] = n  # 存储结果
                break  # 找到满足条件的最小n后退出循环
    # 准备数据
    ratios_list = list(results.keys())
    min_n_list = list(results.values())  # n的绝对值
    min_n_percent_list = [n / N * 100 for n in min_n_list]  # n作为N的百分比
    return ratios_list, min_n_list, min_n_percent_list

# 计算 CV = 0.1 的最小n和kn_ratio_product
ratios_list_1, min_n_list_1, min_n_percent_list_1 = calculate_min_n_and_kn_ratio_product(CV_1, ratios, N)
kn_ratio_product_1 = [r * k for r, k in zip(ratios_list_1, min_n_list_1)]

# 计算 CV = 0.05 的最小n和kn_ratio_product
ratios_list_2, min_n_list_2, min_n_percent_list_2 = calculate_min_n_and_kn_ratio_product(CV_2, ratios, N)
kn_ratio_product_2 = [r * k for r, k in zip(ratios_list_2, min_n_list_2)]

# 创建图和轴
fig, ax1 = plt.subplots(figsize=(8, 8))

# 绘制CV=0.1的最小n/N百分比与Positive %的关系
ratios_percentage1 = [r * 100 for r in ratios_list_1]  # 转换为百分比
ax1.plot(ratios_percentage1, min_n_percent_list_1, '-o', color='tab:blue', label=f'Minimum n/N (%) - CV={CV_1*100}%')

# 绘制CV=0.05的最小n/N百分比与Positive %的关系
ratios_percentage2 = [r * 100 for r in ratios_list_2]  # 转换为百分比

ax1.plot(ratios_percentage2, min_n_percent_list_2, '-o', color='tab:orange', label=f'Minimum n/N (%) - CV={CV_2*100}%')

ax1.set_xlabel('K/N%', fontsize=25)
plt.xticks(fontsize=20)
# 设置x轴为对数尺度
ax1.set_xscale('log')
ax1.set_ylabel('Minimum n/N (%)', color='black', fontsize=25)
ax1.tick_params(axis='y', labelcolor='black')
plt.yticks(fontsize=20)

# 创建共享x轴的另一个轴对象
ax2 = ax1.twinx()

# 绘制CV=0.1的Positive % * Minimum n的关系（使用原始n值）

ax2.plot(ratios_percentage1, kn_ratio_product_1, '-s', color='tab:red', label=f'Minimum k - CV={CV_1*100}%')

# 绘制CV=0.05的Positive % * Minimum n的关系（使用原始n值）
ax2.plot(ratios_percentage2, kn_ratio_product_2, '-s', color='tab:green', label=f'Minimum k - CV={CV_2*100}%')

ax2.set_ylabel('Minimum k', color='black', fontsize=25)
ax2.tick_params(axis='y', labelcolor='black')

# 绘制λ1, λ2, λ2的垂直线
ax1.axvline(x=x1, color='gray', linestyle='--', label=f'λ1 = 131/N')
# ax1.axvline(x=x2, color='black', linestyle='--', label=f'λ2 = {λ2}')
ax1.axvline(x=x3, color='green', linestyle='--', label=f'λ2 = {λ2}')

# 图表标题
plt.title(f'Minimum n/N (%) vs K/N%\nMinimum k vs K/N%\nN={N} at CV={CV_1 * 100}% and {CV_2 * 100}%', fontsize=20)

# 调整布局
dpi_value = 300
fig.tight_layout()  # 调整布局

# # 显示图例
# ax1.legend(loc='upper left', fontsize=12)
# ax2.legend(loc='upper right', fontsize=12)


fig.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust the bottom spacing

# Place legends outside
ax1.legend(loc='upper right', fontsize=15, bbox_to_anchor=(0.55, -0.2), ncol=1)
ax2.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0.55, -0.2), ncol=1)


# 加载CSV文件中的置信区间数据
confidence_intervals = pd.read_csv('updated_confidence_intervals.csv')


# 固定误差线的长度（单位：百分比）
error_length = 4  # 误差线的长度（%），可以调整为你想要的固定高度




# 绘制置信区间
for _, row in confidence_intervals.iterrows():

    
    
    
    
    final_lambda_lower = row['final_lambda_lower']
    final_lambda_upper = row['final_lambda_upper']

    # Apply np.exp to the values
    lower_bound = 1 - np.exp(-final_lambda_lower)
    upper_bound = 1 - np.exp(-final_lambda_upper)
    
    
    
    
    n_over_n = row['n/N']
    
    # 转换为百分比
    lower_bound_percentage = lower_bound * 100
    upper_bound_percentage = upper_bound * 100
    n_over_n_percentage = n_over_n * 100
    
    # 绘制上下误差线（误差线长度固定，但y轴位置根据n/N数据变化）
    ax1.plot([lower_bound_percentage, lower_bound_percentage], [n_over_n_percentage - error_length / 2, n_over_n_percentage + error_length / 2], color='black', linestyle='-')
    ax1.plot([upper_bound_percentage, upper_bound_percentage], [n_over_n_percentage - error_length / 2, n_over_n_percentage + error_length / 2], color='black', linestyle='-')
    
    # 绘制连接上下误差线的横线
    ax1.plot([lower_bound_percentage, upper_bound_percentage], [n_over_n_percentage, n_over_n_percentage], color='black', linestyle='-',alpha=0.8)






# 第二组点（黑色）注意这里是200万磁珠的磁珠计数法
points_x2 = [ 
0.4926274300 ,
0.1206349861,
0.0134715251 ,
0.0021855169 ,
0.0005919892 ,
0.0005782859 ,
0.0004986182 ,
0.0003624502 
]  # 示例第二组点的x坐标
points_y2 = [
9.15895,
9.69785,
16.10805,
13.29205,
16.13205,
13.22875,
16.44545,
20.00275
]  # 示例第二组点的y坐标
ax1.plot([x * 100 for x in points_x2], points_y2, 'cx', markersize=10, alpha=1)  # 蓝色方形，大小8，透明度0.6



plt.yticks(fontsize=20)

dpi_value = 300
fig.tight_layout()  # 调整布局

plt.savefig('图七最小的nN比值和k.png', dpi=dpi_value)  # 保存图像
# 显示图表
plt.show()















# # -*- coding: utf-8 -*-  
# """Created on Sun Nov 17 17:18:36 2024
# @author: lihua"""

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # 总体大小
# N = 500000
# λ1 = 131 / N
# λ2 = 0.2071
# λ2 = 1.2040

# # 计算λ对应的x值
# x1 = (1 - np.exp(-λ1))*100
# x2 = (1 - np.exp(-λ2))*100
# x3 = (1 - np.exp(-λ2))*100

# # 初始CV = 0.1
# CV_1 = 0.1
# # 新增 CV = 0.05
# CV_2 = 0.05
# # 使用对数分布生成 ratios
# ratios = np.logspace(np.log10(1 / N), np.log10((N - 1) / N), 20)

# # 计算第一个CV值对应的最小n和kn_ratio_product
# def calculate_min_n_and_kn_ratio_product(CV, ratios, N):
#     results = {}
#     for ratio in ratios:
#         K = int(N * ratio)  # 根据比值计算K
#         # 寻找满足条件的最小n值
#         for n in range(1, N + 1):
#             left_side = CV * n * (K / N)  # 均值与CV的乘积
#             right_side = np.sqrt(n * (K / N) * (1 - K / N) * ((N - n) / (N - 1)))  # 标准差
#             if left_side > right_side:
#                 results[ratio] = n  # 存储结果
#                 break  # 找到满足条件的最小n后退出循环
#     # 准备数据
#     ratios_list = list(results.keys())
#     min_n_list = list(results.values())  # n的绝对值
#     min_n_percent_list = [n / N * 100 for n in min_n_list]  # n作为N的百分比
#     return ratios_list, min_n_list, min_n_percent_list

# # 计算 CV = 0.1 的最小n和kn_ratio_product
# ratios_list_1, min_n_list_1, min_n_percent_list_1 = calculate_min_n_and_kn_ratio_product(CV_1, ratios, N)
# kn_ratio_product_1 = [r * k for r, k in zip(ratios_list_1, min_n_list_1)]

# # 计算 CV = 0.05 的最小n和kn_ratio_product
# ratios_list_2, min_n_list_2, min_n_percent_list_2 = calculate_min_n_and_kn_ratio_product(CV_2, ratios, N)
# kn_ratio_product_2 = [r * k for r, k in zip(ratios_list_2, min_n_list_2)]

# # 创建图和轴
# fig, ax1 = plt.subplots(figsize=(8, 8))

# # 绘制CV=0.1的最小n/N百分比与Positive %的关系
# ratios_percentage1 = [r * 100 for r in ratios_list_1]  # 转换为百分比
# ax1.plot(ratios_percentage1, min_n_percent_list_1, '-o', color='tab:blue', label=f'Minimum n/N (%) - CV={CV_1*100}%')

# # 绘制CV=0.05的最小n/N百分比与Positive %的关系
# ratios_percentage2 = [r * 100 for r in ratios_list_2]  # 转换为百分比

# ax1.plot(ratios_percentage2, min_n_percent_list_2, '-o', color='tab:orange', label=f'Minimum n/N (%) - CV={CV_2*100}%')

# ax1.set_xlabel('K/N%', fontsize=25)
# plt.xticks(fontsize=20)
# # 设置x轴为对数尺度
# ax1.set_xscale('log')
# ax1.set_ylabel('Minimum n/N (%)', color='black', fontsize=25)
# ax1.tick_params(axis='y', labelcolor='black', labelsize=25)
# plt.yticks(fontsize=20)

# # 创建共享x轴的另一个轴对象
# ax2 = ax1.twinx()

# # 绘制CV=0.1的Positive % * Minimum n的关系（使用原始n值）

# ax2.plot(ratios_percentage1, kn_ratio_product_1, '-s', color='tab:red', label=f'Minimum k - CV={CV_1*100}%')

# # 绘制CV=0.05的Positive % * Minimum n的关系（使用原始n值）
# ax2.plot(ratios_percentage2, kn_ratio_product_2, '-s', color='tab:green', label=f'Minimum k - CV={CV_2*100}%')

# ax2.set_ylabel('Minimum k', color='black', fontsize=25)
# ax2.tick_params(axis='y', labelcolor='black', labelsize=2)

# # 绘制λ1, λ2, λ2的垂直线
# ax1.axvline(x=x1, color='gray', linestyle='--', label=f'λ1 = 131/N')
# # ax1.axvline(x=x2, color='black', linestyle='--', label=f'λ2 = {λ2}')
# ax1.axvline(x=x3, color='green', linestyle='--', label=f'λ2 = {λ2}')

# # 图表标题
# plt.title(f'Minimum n/N (%) vs K/N%\nMinimum k vs K/N%\nN={N} at CV={CV_1 * 100}% and {CV_2 * 100}%', fontsize=20)

# # 调整布局
# dpi_value = 300
# fig.tight_layout()  # 调整布局

# # # 显示图例
# # ax1.legend(loc='upper left', fontsize=12)
# # ax2.legend(loc='upper right', fontsize=12)


# fig.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust the bottom spacing

# # Place legends outside
# ax1.legend(loc='upper right', fontsize=15, bbox_to_anchor=(0.55, -0.2), ncol=1)
# ax2.legend(loc='upper left', fontsize=15, bbox_to_anchor=(0.55, -0.2), ncol=1)


# # 加载CSV文件中的置信区间数据
# confidence_intervals = pd.read_csv('updated_confidence_intervals.csv')


# # 固定误差线的长度（单位：百分比）
# error_length = 4  # 误差线的长度（%），可以调整为你想要的固定高度

# # 绘制置信区间
# for _, row in confidence_intervals.iterrows():

    
    
    
    
#     final_lambda_lower = row['final_lambda_lower']
#     final_lambda_upper = row['final_lambda_upper']

#     # Apply np.exp to the values
#     lower_bound = 1 - np.exp(-final_lambda_lower)
#     upper_bound = 1 - np.exp(-final_lambda_upper)
    
    
    
    
#     n_over_n = row['n/N']
    
#     # 转换为百分比
#     lower_bound_percentage = lower_bound * 100
#     upper_bound_percentage = upper_bound * 100
#     n_over_n_percentage = n_over_n * 100
    
#     # 绘制上下误差线（误差线长度固定，但y轴位置根据n/N数据变化）
#     ax1.plot([lower_bound_percentage, lower_bound_percentage], [n_over_n_percentage - error_length / 2, n_over_n_percentage + error_length / 2], color='black', linestyle='-')
#     ax1.plot([upper_bound_percentage, upper_bound_percentage], [n_over_n_percentage - error_length / 2, n_over_n_percentage + error_length / 2], color='black', linestyle='-')
    
#     # 绘制连接上下误差线的横线
#     ax1.plot([lower_bound_percentage, upper_bound_percentage], [n_over_n_percentage, n_over_n_percentage], color='black', linestyle='-',alpha=0.8)






# # 保存图像时指定dpi

# points_x1 = [
# 0.062430311,
# 0.006766655,
# 0.00089457,
# 0.337351797,
# 0.236263696,
# 0.112863373,
# 0.005316289,
# 0.148079577,
# 0.014918634,
# 0.027328113,
# 0.051968782,
# 0.031787344,
# 0.009570137,
# 0.065990297,
# 0.016807213,

# ]  # x 坐标
# points_y1 = [
# 18.1162,
# 22.936,
# 27.9464,
# 26.1128,
# 27.289,
# 23.0686,
# 33.1058,
# 23.7656,
# 34.5474,
# 28.5274,
# 39.593,
# 34.196,
# 39.059,
# 40.6484,
# 34.271,

# ]  # y 坐标
# # 第二组点（黑色）注意这里是50万磁珠是simoa的数据
# points_x3 = [ 
# # 0.003668466,
# # 0.00290855,
# # 0.026407112,
# # 0.0252009,
# # 0.063376928,
# # 0.066435759,
# # 0.165292823,
# # 0.17879348,
# # 0.441574617,
# # 0.418787248,
# # 0.806804296,
# # 0.825476862,
# # 0.988428745,
# # 0.992390449,
# # 0.999965229,
# # 0.999964689,
# # 0.210691067,
# # 0.238478157,
# # 0.926980894,
# # 0.923003936,
# # 0.05653626,
# # 0.056739596,
# # 0.296232945,
# # 0.309218537,
# # 0.097618943,
# # 0.100646269,
# # 0.118531163,
# # 0.118017036,
# # 0.286657024,
# # 0.288923386,
# # 0.089185338,
# # 0.08879757,
# # 0.908894777,
# # 0.896957885,
# # 0.286330518,
# # 0.295161972,
# # 0.999946521,
# # 0.999946561,
# # 0.341225022,
# # 0.343882796,
# # 0.999940632,
# # 0.999938146,
# # 0.968384366,
# # 0.971929997,
# # 0.997079226,
# # 0.998194312,
# # 0.999914325,
# # 0.999916708,
# # 0.999004276,
# # 0.998891223,
# # 0.865958993,
# # 0.896956582,
# # 0.999959868,
# # 0.999958578,
# # 0.979987088,
# # 0.985366706,
# # 0.999743634,
# # 0.99988672,
# # 0.99953919,
# # 0.999583135,
# # 0.517671775,
# # 0.524769475,
# # 0.664377269,
# # 0.686187545,
# # 0.316841056,
# # 0.30257179,
# # 0.259535312,
# # 0.266110322,
# # 0.999925057,
# # 0.999935982,
# # 0.973860616,
# # 0.978007946,
# # 0.730833926,
# # 0.717942307,
# # 0.388065523,
# # 0.38733617,
# # 0.151820815,
# # 0.149240104,
# # 0.058544304,
# # 0.059557169,
# # 0.022172336,
# # 0.020795752,
# # 0.004037782,
# # 0.003902439



# 0.003668466,
# 0.00290855,
# 0.026407112,
# 0.0252009,
# 0.063376928,
# 0.066435759,
# 0.165292823,
# 0.17879348,
# 0.441574617,
# 0.418787248,
# 0.806804296,
# 0.825476862,
# 0.988428745,
# 0.992390449,
# 0.999965229,
# 0.999964689,
# 0.210691067,
# 0.238478157,
# 0.926980894,
# 0.923003936,
# 0.999946521,
# 0.999946561,
# 0.341225022,
# 0.343882796,
# 0.999940632,
# 0.999938146,
# 0.968384366,
# 0.971929997,
# 0.997079226,
# 0.998194312,
# 0.999914325,
# 0.999916708,
# 0.999004276,
# 0.998891223,
# 0.865958993,
# 0.896956582,
# 0.999959868,
# 0.999958578,
# 0.979987088,
# 0.985366706



# ]  # 示例第二组点的x坐标
# points_y3 = [
# # 6.7058,
# # 5.9136,
# # 6.0514,
# # 6.222,
# # 5.4594,
# # 6.6982,
# # 4.6342,
# # 7.1412,
# # 6.1018,
# # 6.3426,
# # 6.2196,
# # 6.029,
# # 4.5976,
# # 4.5732,
# # 5.752,
# # 5.664,
# # 5.8142,
# # 6.6656,
# # 5.6944,
# # 5.6912,
# # 4.192,
# # 4.9454,
# # 5.0278,
# # 5.5844,
# # 4.5442,
# # 4.4254,
# # 4.8038,
# # 4.0384,
# # 4.4248,
# # 4.3804,
# # 5.0524,
# # 5.597,
# # 5.284,
# # 4.7204,
# # 5.5174,
# # 5.2046,
# # 3.7398,
# # 3.7426,
# # 4.604,
# # 4.9418,
# # 3.3688,
# # 3.2334,
# # 4.017,
# # 4.0684,
# # 3.8346,
# # 3.9874,
# # 4.6688,
# # 4.8024,
# # 3.4146,
# # 3.4272,
# # 4.9552,
# # 4.3438,
# # 4.9836,
# # 4.8284,
# # 4.9568,
# # 4.5786,
# # 4.6808,
# # 5.2966,
# # 4.7742,
# # 3.3584,
# # 4.8552,
# # 6.29,
# # 4.5718,
# # 4.4192,
# # 4.014,
# # 4.2694,
# # 4.347,
# # 5.7758,
# # 5.3374,
# # 6.2482,
# # 5.0422,
# # 5.6384,
# # 6.1828,
# # 6.3994,
# # 6.1536,
# # 5.875,
# # 3.8664,
# # 6.7904,
# # 5.4352,
# # 5.0674,
# # 5.791,
# # 5.8762,
# # 5.5476,
# # 6.355

#  6.71 ,
#  5.91 ,
#  6.05 ,
#  6.22 ,
#  5.46 ,
#  6.70 ,
#  4.63 ,
#  7.14 ,
#  6.10 ,
#  6.34 ,
#  6.22 ,
#  6.03 ,
#  4.60 ,
#  4.57 ,
#  5.75 ,
#  5.66 ,
#  5.81 ,
#  6.67 ,
#  5.69 ,
#  5.69 ,
#  3.74 ,
#  3.74 ,
#  4.60 ,
#  4.94 ,
#  3.37 ,
#  3.23 ,
#  4.02 ,
#  4.07 ,
#  3.83 ,
#  3.99 ,
#  4.67 ,
#  4.80 ,
#  3.41 ,
#  3.43 ,
#  4.96 ,
#  4.34 ,
#  4.98 ,
#  4.83 ,
#  4.96 ,
#  4.58 








# ]  # 示例第二组点的y坐标


# # # 第二组点（黑色）注意这里是200万磁珠的磁珠计数法
# # points_x2 = [ 
# # 0.4926274300 ,
# # 0.1206349861,
# # 0.0134715251 ,
# # 0.0021855169 ,
# # 0.0005919892 ,
# # 0.0005782859 ,
# # 0.0004986182 ,
# # 0.0003624502 
# # ]  # 示例第二组点的x坐标
# # points_y2 = [
# # 9.15895,
# # 9.69785,
# # 16.10805,
# # 13.29205,
# # 16.13205,
# # 13.22875,
# # 16.44545,
# # 20.00275
# # ]  # 示例第二组点的y坐标
# # ax1.plot(points_x2, points_y2, 'gx') 

# # # # 第二组点（黑色）注意这里是40万磁珠的simoa的文献中的原始数据
# # # points_x4 = [ 
# # #  0.000016000000000, 
# # #  0.000086000000000, 
# # #  0.000099000000000,
# # #  0.000413000000000 ,
# # #  0.000713000000000 ,
# # #  0.004461000000000 ,
# # #  0.008183000000000 ,
# # #  0.033802000000000 ,
# # #  0.075865000000000 ,
# # #  0.306479000000000 ,
# # #  0.445296000000000 

# # # ]  # 示例第二组点的x坐标
# # # points_y4 = [
# # # 15.625,
# # # 8.720930233,
# # # 12.62626263,
# # # 13.31719128,
# # # 13.32398317,
# # # 13.28177539,
# # # 11.76218991,
# # # 13.21667357,
# # # 13.29994068,
# # # 12.75291292,
# # # 13.94353419

# # # ]  # 示例第二组点的y坐标
# # # ax1.plot(points_x4, points_y4, 'go') 



# # ax1.plot([x * 100 for x in points_x1], points_y1, 'cs')  # 将x坐标乘以100以显示为百分比
# # ax1.plot([x * 100 for x in points_x3], points_y3, 'ms')  # 将x坐标乘以100以显示为百分比


# # 绘制带有透明度、大小和形状的点
# ax1.plot([x * 100 for x in points_x1], points_y1, 'cx', markersize=5, alpha=1)  # 蓝色方形，大小8，透明度0.6
# ax1.plot([x * 100 for x in points_x3], points_y3, 'mx', markersize=5, alpha=1)  # 品红色方形，大小8，透明度0.6



# plt.savefig('图七最小的nN比值和k.png', dpi=dpi_value)  # 保存图像

# plt.yticks(fontsize=15)
# # 显示图形
# plt.show()












