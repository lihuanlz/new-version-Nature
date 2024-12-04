# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:03:00 2024

@author: lihua
"""

# -*- coding: utf-8 -*-
"""Created on Tue Nov  5 21:27:01 2024
@author: lihua"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 参数组合
k_list = [
    [10000, 0.001, 10000, 0.001, 10000, 0.001, 10000, 0.001],
    [10000, 0.0001, 10000, 0.0001, 10000, 0.0001, 10000, 0.0001],
    [10000, 0.00001, 10000, 0.00001, 10000, 0.00001, 10000, 0.00001],
    [100000, 0.001, 100000, 0.001, 100000, 0.001, 100000, 0.001],
    [100000, 0.0001, 100000, 0.0001, 100000, 0.0001, 100000, 0.0001],
    [100000, 0.00001, 100000, 0.00001, 100000, 0.00001, 100000, 0.00001],
    [1000000, 0.001, 1000000, 0.001, 1000000, 0.001, 1000000, 0.001],
    [1000000, 0.0001, 1000000, 0.0001, 1000000, 0.0001, 1000000, 0.0001],
    [1000000, 0.00001, 1000000, 0.00001, 1000000, 0.00001, 1000000, 0.00001]
]

# 初始条件
x0 = [0, 0, 0]

# 时间跨度
t = np.linspace(0, 7200, 50000)  # 从0到3600，共50000点

# 微分方程
def model(t, x, k):
    x1, x2, x3 = x
    k1, k2, k3, k4, k5, k6, k7, k8 = k
    
    # 通过约束计算x4, x5, x6
    x4 = 1E-18 - (x1 + x2 + x3)
    x5 = 1E-9 - (x1 + x2)
    x6 = 1E-9 - (x1 + x3)
    
    # 微分方程
    dx1dt = k3*x2*x6 + k7*x3*x5 - (k4*x1 + k8*x1)
    dx2dt = k1*x4*x5 + k4*x1 - (k2*x2 + k3*x2*x6)
    dx3dt = k5*x4*x6 + k8*x1 - (k6*x3 + k7*x3*x5)
    
    return [dx1dt, dx2dt, dx3dt]

# 绘制比较图
plt.figure(figsize=(8, 8))
initial_x4 = 1E-18  # 初始x4浓度
legend_info = []

for k in k_list:
    # 求解ODE
    sol = solve_ivp(model, [t[0], t[-1]], x0, t_eval=t, args=(k,), rtol=1e-60, atol=1e-60)
    
    # 计算x1与初始x4浓度的百分比
    x1_percentage = (sol.y[0] / initial_x4) * 100
    
    # 绘制x1百分比随时间变化的图
    plt.plot(sol.t / 60, x1_percentage, linewidth=2)  # Divide time by 60 to convert seconds to minutes
    legend_info.append(f'kon={k[0]:.1e}, koff={k[1]:.1e}')

plt.xlabel('Time (min)', fontsize=15)  # Change the label to 'min'
plt.ylabel('Complex as % of initial Ag concentration', fontsize=15)
plt.title('Change of complex as Percentage of Initial Ag Concentration over Time', fontsize=15)
plt.legend(legend_info, loc='best', fontsize=10)
# plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Save the plot with 300 dpi
dpi_value = 300
plt.savefig('Kinetics_minutes.png', dpi=dpi_value)
filename = f'plot of dy2.png'
plt.savefig(filename, dpi=300)
# Show the plot
plt.show()
