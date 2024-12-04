
import numpy as np
import matplotlib.pyplot as plt

# 定义参数

N = 2000000
aglost = 0.1
beadlost = 0.1
M_max = -N * np.log(1 / N)
K_max = N * (1 - np.exp(-M_max / N))
left = 1 - beadlost
agleft=1- aglost
# 设置n的范围
n_values = range(1, 13)  # n 从 1 到 13

# 开始绘图
plt.figure(figsize=(8, 8))

# 对于每个n值计算K并绘图
for n in n_values:
    # 设置 M_values 的范围为 0 到 N * (1 - np.exp(-M_max / N)) * left**(n-1)
    M_max_n = -N * np.log(1 / N) * agleft**(n-1)
    if M_max_n > 0:
        M_values = np.logspace(np.log10(0.1), np.log10(M_max_n), 10000)  # 计算 M 从 0.1 到 M_max_n 的值 (避开0)
    else:
        M_values = np.array([0.1])  # 如果 M_max_n 不大于 0, 则设置 M_values 为一个近似于 0 的值
    K_values = N * left**(n-1) * (1 - np.exp((-M_values) / ((N * left**(n-1)))))  # 计算每个n的K值
    plt.plot(M_values, K_values, label=f'n={n}')  # 绘制每个n的曲线

# 标记和注释 n=1 的曲线上的点
n = 1
M_values_n1 = [1.2040 * N]
for M in M_values_n1:
    K_value = N * left**(n-1) * (1 - np.exp(-M / (N * left**(n-1))))  # 计算K值
    plt.scatter(M, K_value, color='red')  # 标记点
    plt.annotate(f'M={M:.0f}', (M, K_value), textcoords="offset points", xytext=(-70,-10), ha='center', fontsize=15, color='red')

# 标记和注释 n=12 的曲线上的点
n = 13
M_values_n12 = [int(131 * agleft**(n-1)), int(1.2040 * N * agleft**(n-1))]
for M in M_values_n12:
    K_value = N * left**(n-1) * (1 - np.exp(-M / (N * left**(n-1))))  # 计算K值
    plt.scatter(M, K_value, color='blue')  # 标记点
    plt.annotate(f'M={M:.0f}', (M, K_value), textcoords="offset points", xytext=(40,-30), ha='center', fontsize=15, color='blue')

# 图表设置
plt.xlabel('M', fontsize=15)
plt.ylabel('K', fontsize=15)
plt.title(f'Relationship between M and K\n Initial N={N}', fontsize=15)
plt.legend(title='Wash times', fontsize=15)

plt.xscale('log')
plt.yscale('log')
#plt.xlim(0.1, M_max)  # 设置 x 轴的范围
#plt.ylim(1, K_max)   # 设置 y 轴的范围

plt.show()
