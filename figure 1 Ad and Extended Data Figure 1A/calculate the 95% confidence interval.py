
'''
这里注意 拷贝原始的N n 以及K的信息即可计算

'''




# import numpy as np
# from scipy.stats import hypergeom
# import pandas as pd

# # 定义函数来计算95%置信区间
# def calculate_confidence_interval(N, n, k):
#     # K 的可能取值范围
#     K_values = np.arange(k, N - (n - k) + 1)
    
#     # 计算每个 K 值的概率
#     pmf_values = hypergeom.pmf(k, N, K_values, n)
    
#     # 归一化概率
#     pmf_values /= pmf_values.sum()
    
#     # 计算累积概率
#     cdf_values = np.cumsum(pmf_values)
    
#     # 找到置信区间
#     lower_bound = K_values[np.searchsorted(cdf_values, 0.025)]
#     upper_bound = K_values[np.searchsorted(cdf_values, 0.975)]
    
#     return lower_bound, upper_bound

# # 用户输入的 N, n, k 数值
# N_values = [500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000

# ]
# n_values = [33529,
# 29568,
# 30257,
# 31110,
# 27297,
# 33491,
# 23171,
# 35706,
# 30509,
# 31713,
# 31098,
# 30145,
# 22988,
# 22866,
# 28760,
# 28320,
# 29071,
# 33328,
# 28472,
# 28456,
# 18699,
# 18713,
# 23020,
# 24709,
# 16844,
# 16167,
# 20085,
# 20342,
# 19173,
# 19937,
# 23344,
# 24012,
# 17073,
# 17136,
# 24776,
# 21719,
# 24918,
# 24142,
# 24784,
# 22893

# ]
# k_values = [124,
# 86,
# 799,
# 784,
# 1730,
# 2225,
# 3830,
# 6385,
# 13472,
# 13281,
# 25090.14257,
# 24884.3923,
# 22722.50499,
# 22692.03226,
# 28759.87897,
# 28319.95142,
# 6125,
# 7948,
# 26393.31883,
# 26265.96436,
# 18698.999,
# 18712.99851,
# 7855,
# 8497,
# 16843.99982,
# 16166.99989,
# 19450.31655,
# 19771.17306,
# 19117.63287,
# 19901.11468,
# 23342.92198,
# 24010.88113,
# 17056.68663,
# 17117.65866,
# 21455.86666,
# 19481.52687,
# 24917.74005,
# 24141.7941,
# 24288.59087,
# 22558.19784

# ]

# # 确保输入的列表长度一致
# assert len(N_values) == len(n_values) == len(k_values), "N, n, k 列表的长度不一致！"

# # 初始化一个空的列表来存储结果
# results = []

# # 对每组数据计算置信区间
# for N, n, k in zip(N_values, n_values, k_values):
#     lower_bound, upper_bound = calculate_confidence_interval(N, n, k)
#     results.append((N, n, k, lower_bound, upper_bound))
    
#     # 打印每次的计算结果
#     print(f"N={N}, n={n}, k={k} => 95% 置信区间为 [{lower_bound}, {upper_bound}]")

# # 将结果转换为 DataFrame
# df = pd.DataFrame(results, columns=["N", "n", "k", "Lower Bound", "Upper Bound"])

# # 计算 n/N, Lower Bound/N, Upper Bound/N
# df["n/N"] = df["n"] / df["N"]
# df["Lower Bound/N"] = df["Lower Bound"] / df["N"]
# df["Upper Bound/N"] = df["Upper Bound"] / df["N"]

# # 保存到 CSV 文件
# df.to_csv("confidence_intervals_with_ratios.csv", index=False)

# print("计算完成，结果已保存到 'confidence_intervals_with_ratios.csv' 文件中。")






# import numpy as np
# from scipy.stats import hypergeom
# import pandas as pd

# # 定义函数来计算95%置信区间
# def calculate_confidence_interval(N, n, k):
#     # K 的可能取值范围
#     K_values = np.arange(k, N - (n - k) + 1)
    
#     # 计算每个 K 值的概率
#     pmf_values = hypergeom.pmf(k, N, K_values, n)
    
#     # 归一化概率
#     pmf_values /= pmf_values.sum()
    
#     # 计算累积概率
#     cdf_values = np.cumsum(pmf_values)
    
#     # 找到置信区间
#     lower_bound = K_values[np.searchsorted(cdf_values, 0.025)]
#     upper_bound = K_values[np.searchsorted(cdf_values, 0.975)]
    
#     return lower_bound, upper_bound

# # 用户输入的 N, n, k 数值
# N_values = [400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000,
# 400000

# ]
# n_values = [62500,
# 34884 ,
# 50505 ,
# 53269 ,
# 53296 ,
# 53127 ,
# 47049 ,
# 52867 ,
# 53200 ,
# 51012 ,
# 55774 

# ]
# k_values = [1,
# 3,
# 5,
# 22,
# 38,
# 237,
# 385,
# 1787,
# 4036,
# 15634,
# 24836

# ]

# # 确保输入的列表长度一致
# assert len(N_values) == len(n_values) == len(k_values), "N, n, k 列表的长度不一致！"

# # 初始化一个空的列表来存储结果
# results = []

# # 对每组数据计算置信区间
# for N, n, k in zip(N_values, n_values, k_values):
#     lower_bound, upper_bound = calculate_confidence_interval(N, n, k)
#     results.append((N, n, k, lower_bound, upper_bound))
    
#     # 打印每次的计算结果
#     print(f"N={N}, n={n}, k={k} => 95% 置信区间为 [{lower_bound}, {upper_bound}]")

# # 将结果转换为 DataFrame
# df = pd.DataFrame(results, columns=["N", "n", "k", "Lower Bound", "Upper Bound"])

# # 计算 n/N, Lower Bound/N, Upper Bound/N
# df["n/N"] = df["n"] / df["N"]
# df["Lower Bound/N"] = df["Lower Bound"] / df["N"]
# df["Upper Bound/N"] = df["Upper Bound"] / df["N"]

# # 保存到 CSV 文件
# df.to_csv("confidence_intervals_with_ratios.csv", index=False)

# print("计算完成，结果已保存到 'confidence_intervals_with_ratios.csv' 文件中。")





import numpy as np
from scipy.stats import hypergeom
import pandas as pd

# 定义函数来计算95%置信区间
def calculate_confidence_interval(N, n, k):
    # K 的可能取值范围
    K_values = np.arange(k, N - (n - k) + 1)
    
    # 计算每个 K 值的概率
    pmf_values = hypergeom.pmf(k, N, K_values, n)
    
    # 归一化概率
    pmf_values /= pmf_values.sum()
    
    # 计算累积概率
    cdf_values = np.cumsum(pmf_values)
    
    # 找到置信区间
    lower_bound = K_values[np.searchsorted(cdf_values, 0.025)]
    upper_bound = K_values[np.searchsorted(cdf_values, 0.975)]
    
    return lower_bound, upper_bound

# 用户输入的 N, n, k 数值
N_values = [2000000,
2000000,
2000000,
2000000,
2000000,
2000000,
2000000,
2000000
]
n_values = [183179,
193957,
322161,
265841,
322641,
264575,
328909,
400055
]
k_values = [90239,
23398,
4340,
581,
191,
153,
164,
145
]

# 确保输入的列表长度一致
assert len(N_values) == len(n_values) == len(k_values), "N, n, k 列表的长度不一致！"

# 初始化一个空的列表来存储结果
results = []

# 对每组数据计算置信区间
for N, n, k in zip(N_values, n_values, k_values):
    lower_bound, upper_bound = calculate_confidence_interval(N, n, k)
    results.append((N, n, k, lower_bound, upper_bound))
    
    # 打印每次的计算结果
    print(f"N={N}, n={n}, k={k} => 95% 置信区间为 [{lower_bound}, {upper_bound}]")

# 将结果转换为 DataFrame
df = pd.DataFrame(results, columns=["N", "n", "k", "Lower Bound", "Upper Bound"])

# 计算 n/N, Lower Bound/N, Upper Bound/N
df["n/N"] = df["n"] / df["N"]
df["Lower Bound lamuda"] = -np.log(1-df["Lower Bound"] / df["N"])
df["Upper Bound lamuda"] = -np.log(1-df["Upper Bound"] / df["N"])

# 保存到 CSV 文件
df.to_csv("confidence_intervals_with_ratios.csv", index=False)

print("计算完成，结果已保存到 'confidence_intervals_with_ratios.csv' 文件中。")






# import numpy as np
# from scipy.stats import hypergeom
# import pandas as pd

# # 定义函数来计算95%置信区间
# def calculate_confidence_interval(N, n, k):
#     # K 的可能取值范围
#     K_values = np.arange(k, N - (n - k) + 1)
    
#     # 计算每个 K 值的概率
#     pmf_values = hypergeom.pmf(k, N, K_values, n)
    
#     # 归一化概率
#     pmf_values /= pmf_values.sum()
    
#     # 计算累积概率
#     cdf_values = np.cumsum(pmf_values)
    
#     # 找到置信区间
#     lower_bound = K_values[np.searchsorted(cdf_values, 0.025)]
#     upper_bound = K_values[np.searchsorted(cdf_values, 0.975)]
    
#     return lower_bound, upper_bound

# # 用户输入的 N, n, k 数值
# N_values = [
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,

# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000,
# 500000




# ]
# n_values = [
# 90581,
# 114680,
# 139732,
# 130564,
# 136445,
# 115343,
# 165529,
# 118828,
# 172737,
# 142637,
# 197965,
# 170980,
# 195295,
# 203242,
# 171355,
  
    
# 33529,
# 29568,
# 30257,
# 31110,
# 27297,
# 33491,
# 23171,
# 35706,
# 30509,
# 31713,
# 31098,
# 30145,
# 22988,
# 22866,
# 28760,
# 28320,
# 29071,
# 33328,
# 28472,
# 28456,
# 18699,
# 18713,
# 23020,
# 24709,
# 16844,
# 16167,
# 20085,
# 20342,
# 19173,
# 19937,
# 23344,
# 24012,
# 17073,
# 17136,
# 24776,
# 21719,
# 24918,
# 24142,
# 24784,
# 22893




# ]
# k_values = [
# 5655,
# 776,
# 125,
# 44046,
# 32237,
# 13018,
# 880,
# 17596,
# 2577,
# 3898,
# 10288,
# 5435,
# 1869,
# 13412,
# 2880,
    

# 123,
# 86,
# 799,
# 784,
# 1730,
# 2225,
# 3830,
# 6384,
# 13472,
# 13281,
# 25090,
# 24884,
# 22722,
# 22692,
# 28759,
# 28319,
# 6125,
# 7948,
# 26393,
# 26265,
# 18698,
# 18712,
# 7855,
# 8497,
# 16843,
# 16166,
# 19450,
# 19771,
# 19117,
# 19901,
# 23342,
# 24010,
# 17056,
# 17117,
# 21455,
# 19481,
# 24917,
# 24141,
# 24288,
# 22558




# ]

# # 确保输入的列表长度一致
# assert len(N_values) == len(n_values) == len(k_values), "N, n, k 列表的长度不一致！"

# # 初始化一个空的列表来存储结果
# results = []

# # 对每组数据计算置信区间
# for N, n, k in zip(N_values, n_values, k_values):
#     lower_bound, upper_bound = calculate_confidence_interval(N, n, k)
#     results.append((N, n, k, lower_bound, upper_bound))
    
#     # 打印每次的计算结果
#     print(f"N={N}, n={n}, k={k} => 95% 置信区间为 [{lower_bound}, {upper_bound}]")

# # 将结果转换为 DataFrame
# df = pd.DataFrame(results, columns=["N", "n", "k", "Lower Bound", "Upper Bound"])

# # 计算 n/N, Lower Bound/N, Upper Bound/N
# df["n/N"] = df["n"] / df["N"]
# df["Lower Bound lamuda"] = -np.log(1-df["Lower Bound"] / df["N"])
# df["Upper Bound lamuda"] = -np.log(1-df["Upper Bound"] / df["N"])

# # 保存到 CSV 文件
# df.to_csv("confidence_intervals_with_ratios.csv", index=False)

# print("计算完成，结果已保存到 'confidence_intervals_with_ratios.csv' 文件中。")











































from scipy.stats import poisson
from scipy.stats import chi2

# 读取CSV文件
df = pd.read_csv('confidence_intervals_with_ratios.csv')

# 设置显著性水平 alpha
alpha = 0.05        # 显著性水平，95% 置信区间

# 用于存储计算结果的列表
final_lambda_lower_list = []
final_lambda_upper_list = []

# 遍历每一行数据，计算 lambda 的置信区间
for index, row in df.iterrows():
    # 从数据行中获取样本大小 N
    N = row['N']  # 假设 CSV 文件中的第一列是 'Sample Size'
    
    lower_bound_lambda = row['Lower Bound lamuda']
    upper_bound_lambda = row['Upper Bound lamuda']

    # 计算对应的 observed_events
    lower_bound_events = lower_bound_lambda * N
    upper_bound_events = upper_bound_lambda * N

    # 使用 Poisson 分布的反向累积分布函数计算 Lower Bound λ 的 95% 置信区间
    # lower_bound_lambda_lower = poisson.ppf(alpha / 2, lower_bound_events) / N
    # lower_bound_lambda_upper = poisson.ppf(1 - alpha / 2, lower_bound_events) / N
    
    
    
    lower_bound_lambda_lower = chi2.ppf(alpha / 2, 2 * lower_bound_events) / (2 * N)
    lower_bound_lambda_upper = chi2.ppf(1 - alpha / 2, 2 * (lower_bound_events + 1)) / (2 * N)
    

    # 使用 Poisson 分布的反向累积分布函数计算 Upper Bound λ 的 95% 置信区间
    # upper_bound_lambda_lower = poisson.ppf(alpha / 2, upper_bound_events) / N
    # upper_bound_lambda_upper = poisson.ppf(1 - alpha / 2, upper_bound_events) / N
    
    
    upper_bound_lambda_lower = chi2.ppf(alpha / 2, 2 * upper_bound_events) / (2 * N)
    upper_bound_lambda_upper = chi2.ppf(1 - alpha / 2, 2 * (upper_bound_events + 1)) / (2 * N)
    

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














