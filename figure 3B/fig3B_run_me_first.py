import pandas as pd
import numpy as np

# 定义参数
N = 2000000

error_limit =0.15
Ms_values = np.logspace(np.log10(1), np.log10(N-1), num=100,dtype=int)

N_values = np.logspace(np.log10(1), np.log10(N), num=100,dtype=int)

M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=100)

# Prepare a list to collect all the rows for DataFrame
data_list = []

# 外层循环遍历不同的N值
for N in N_values:
    # 中层循环遍历不同的Ms值
    for Ms in Ms_values:
        # 内层循环遍历不同的M/N ratios
        for M_N_ratio in M_N_ratios:
            C1 = (1 - np.exp(-M_N_ratio)) / (1 - np.exp(-Ms/N))
            C2 = M_N_ratio * N / Ms
            relative_error = np.abs((C1 - C2) / C2)
            # Only add data if the relative error is less than 0.1
            if relative_error < error_limit:
                data_list.append({'Ms': Ms,'N': N,  'M_N_ratio': M_N_ratio, 'relative_error': relative_error})

# Convert the list to DataFrame
results_df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
results_df.to_csv('5.5_relative_error_between_lam_and_Poscent.csv', index=False)


