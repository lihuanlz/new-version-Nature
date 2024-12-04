# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:02:05 2024

@author: lihua
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义速率常数
k_on1, k_off1 = 1e5, 1e-4  # A + B -> AB
k_on2, k_off2 = 1e5, 1e-4  # AB + C -> ABC
k_on3, k_off3 = 5.1e6, 5.1e-9  # ABC + D -> ABCD

# 初始浓度
A0, B0, C0, D0 = 1e-9, 1e-18, 1e-9, 1e-9

# 定义微分方程
def reaction1(t, y):
    A, B, AB = y
    dA_dt = -k_on1 * A * B + k_off1 * AB
    dB_dt = -k_on1 * A * B + k_off1 * AB
    dAB_dt = k_on1 * A * B - k_off1 * AB
    return [dA_dt, dB_dt, dAB_dt]

def reaction2(t, y):
    AB, C, ABC = y
    dAB_dt = -k_on2 * AB * C + k_off2 * ABC
    dC_dt = -k_on2 * AB * C + k_off2 * ABC
    dABC_dt = k_on2 * AB * C - k_off2 * ABC
    return [dAB_dt, dC_dt, dABC_dt]

def reaction3(t, y):
    ABC, D, ABCD = y
    dABC_dt = -k_on3 * ABC * D + k_off3 * ABCD
    dD_dt = -k_on3 * ABC * D + k_off3 * ABCD
    dABCD_dt = k_on3 * ABC * D - k_off3 * ABCD
    return [dABC_dt, dD_dt, dABCD_dt]

# 获取用户输入的时间
time_step1 = 60 * 60
time_step2 = 60 * 60
time_step3 = 10 * 60

# 解第一个反应的方程
y0_1 = [A0, B0, 0.0]
t_eval1 = np.linspace(0, time_step1, 20000)
sol1 = solve_ivp(reaction1, [0, time_step1], y0_1, method='RK45', t_eval=t_eval1, rtol=1e-30, atol=1e-100)
AB_final = sol1.y[2, -1]

# 解第二个反应的方程
y0_2 = [AB_final, C0, 0.0]
t_eval2 = np.linspace(0, time_step2, 20000)
sol2 = solve_ivp(reaction2, [0, time_step2], y0_2, method='RK45', t_eval=t_eval2, rtol=1e-30, atol=1e-100)
ABC_final = sol2.y[2, -1]

# 解第三个反应的方程
y0_3 = [ABC_final, D0, 0.0]
t_eval3 = np.linspace(0, time_step3, 20000)
sol3 = solve_ivp(reaction3, [0, time_step3], y0_3, method='RK45', t_eval=t_eval3, rtol=1e-30, atol=1e-100)
ABCD_final = sol3.y[2, -1]

print(f"final ABCD concentration: {ABCD_final}")

# 绘图
t1 = sol1.t / 60  # 转换为分钟
t2 = sol2.t / 60 + time_step1 / 60  # 转换为分钟并加上第一个时间的偏移
t3 = sol3.t / 60 + (time_step1 + time_step2) / 60  # 转换为分钟并加上前两个时间的偏移

plt.figure(figsize=(8, 8))
plt.plot(t1, (sol1.y[2] / B0) * 100, label='[Ab1Ag] / Ag0 (%)')
plt.plot(t2, (sol2.y[2] / B0) * 100, label='[Ab1AgAb2] / Ag0 (%)')
plt.plot(t3, (sol3.y[2] / B0) * 100, label='[Ab1AgAb2SPA] / Ag0 (%)')
plt.xlabel('Time (min)', fontsize=15)
plt.ylabel('Concentration Ratio (%)', fontsize=15)
plt.legend(fontsize=10)
plt.title('The concentration of reactants changes over time', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

filename = f'plot of dy.png'
plt.savefig(filename, dpi=300)
plt.show()
