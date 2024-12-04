import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
from scipy.special import i0

# 定义参数
N = 500000
Ms = 10000
# a_values = 0.1
# 生成M/N的对数范围值，直接计算M/N的比例，
M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=N) #注意最小的拉姆达为1/N即只有一个分子均匀分布的情况，最大值是对的，最小值注意是1个分子分布到N个磁珠上，这里与阳性要显著区别。。。。。。。。。。。。。。。。。。。。。。

# 初始化满足条件的最大i值
max_postive_beads = 0

# 用于存储计算结果的列表
results = []

# 遍历所有可能的postive_beads值
for M_N_ratio in M_N_ratios:  # 直接迭代M_N_ratios数组
    C1 = (1-np.exp(-M_N_ratio)) / (1-np.exp(-Ms/N))
    # C1 = (1 - np.exp(-M_N_ratio * (1 + a_values)) * i0(2 * M_N_ratio * np.sqrt(a_values))) / (1 - np.exp(-Ms/N * (1 + a_values)) * i0(2 * Ms/N * np.sqrt(a_values)))
    C2 = M_N_ratio * N / Ms
    expression_value = np.abs((C1 - C2) / C2)
    
    # 将结果添加到列表中
    results.append([M_N_ratio, expression_value])

    if expression_value < 0.1:
        max_postive_beads = M_N_ratio * N
    else:
        break

# 写入CSV文件
with open('expression_values.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['M/N ratio', 'Expression Value'])  # 写入头部
    for result in results:
        writer.writerow(result)

print('CSV file has been written with the expression values.')


# 初始化数据存储列表
x_values = []
y_values = []

# 遍历所有可能的 M 值
for M in range(1, int(-N*np.log(1/N)) + 1):
    x = M/N
    y = 1 - np.exp(-M/N)
    x_values.append(x)
    y_values.append(y)

# 主图
plt.figure(figsize=(10, 9))
ax_main = plt.subplot(111)
ax_main.plot(x_values, y_values, linestyle='-', markersize=1, label='(1 - exp(-M/N)) / N vs M/N')
ax_main.set_xlabel('λ=M/N', fontsize=15)
#ax_main.set_xscale('log')#####################################
ax_main.set_ylabel('Postive %', fontsize=15)
ax_main.set_title(f'Postive % vs λ\nN={N},Ms={Ms}', fontsize=15)
max_postive_beads=int(max_postive_beads)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# 标注 max_postive_beads 的点
x_highlight = max_postive_beads / N
y_highlight = (1 - np.exp(-max_postive_beads / N)) 
ax_main.plot(x_highlight, y_highlight, 'ro')
ax_main.annotate(f'M={max_postive_beads}', (x_highlight, y_highlight), textcoords="offset points", xytext=(10,-10), ha='center')


# # 标记特定的x值
x_vals_to_mark = [1.2040, -np.log(1/N)]
labels = [r'$1.2040$', r'$-\ln(1/N)$']
colors = ['black', 'purple']

# 调整 xytext 参数以分开标签
offsets = [(0, 10), (20, 20), (40, 40)]  # 增加不同的水平偏移量

for x_val, label, color in zip(x_vals_to_mark, labels, colors):
    ax_main.axvline(x=x_val, color=color, linestyle='--', label=f'x = {label}')
    ax_main.annotate(f'{label}', (x_val, 0.5), textcoords="offset points", xytext=(0,10), ha='center', color=color)



# 插入小图
ax_inset = inset_axes(ax_main, width="50%", height="50%", loc='center')
x_zoomed = x_values[:max_postive_beads]
y_zoomed = y_values[:max_postive_beads]
ax_inset.plot(x_zoomed, y_zoomed, linestyle='-', color='r')
ax_inset.set_xlabel('λ')
ax_inset.set_ylabel('Postive %')
ax_inset.set_title(f'Postive % vs λ\nMax_postive_beads={max_postive_beads}')

# 在小图上进行线性回归
slope, intercept, r_value, p_value, std_err = linregress(x_zoomed, y_zoomed)
# 计算回归线的y值
y_regress = np.array(x_zoomed) * slope + intercept
# 绘制回归线
ax_inset.plot(x_zoomed, y_regress, 'b--', label='Linear regression')

# 标注回归方程和相关系数
regress_eq = f'y = {slope:.2e}x + {intercept:.2e}\nR^2 = {r_value**2:.2f}'
ax_inset.annotate(regress_eq, xy=(0.5, 0.2), xycoords='axes fraction', ha='center', va='center', fontsize=9, bbox=dict(boxstyle="round", alpha=0.5, color="w"))


dpi_value = 600
plt.legend()

plt.savefig('图十阳性百分比和λ计算.png', dpi=dpi_value)  # 保存图像时指定dpi
plt.show()
