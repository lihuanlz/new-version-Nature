# -*- coding: utf-8 -*-
"""Created on Tue Nov 12 10:25:56 2024
@author: lihuan"""

import numpy as np
import matplotlib.pyplot as plt
import csv  # For saving data to a CSV file

# Define parameters
N = 500000
aglost = 0.1
beadlost = 0.05
agleft = 1 - aglost
left = 1 - beadlost
blank = 50
n_values = range(1, 13)  # n from 1 to 13

# List to store data for CSV output
csv_data = []

# Prepare the figure for plotting
plt.figure(figsize=(8, 8))

# Loop over n values to compute K and M
for n in n_values:
    Nn = N * left ** (n - 1)  # Effective N after n washes
    M_max_n = -N * np.log(1 / N) * agleft ** (n - 1)  # Adjusted M_max_n

    # If M_max_n is positive, calculate M values; otherwise, use a small value
    if M_max_n > 0:
        M_values = np.logspace(0, np.log10(M_max_n), 1000)  # M values from 1 to M_max_n
    else:
        M_values = np.array([0.1])  # Use small value for M if M_max_n <= 0
    
    # Calculate λ and normalized K
    lambda_values = M_values / Nn
    blank_term = blank / Nn
    K_values_modified = 1 - np.exp(- (lambda_values + blank_term))
    lambda_values_fromK = -np.log(1 - K_values_modified)

    # Store the data for each M and corresponding K
    for i in range(len(M_values)):
        csv_data.append([lambda_values[i], K_values_modified[i], Nn, n])

    # Plotting
    plt.plot(lambda_values, K_values_modified, label=f'n={n}')
    plt.plot(lambda_values_fromK, K_values_modified)

# Annotations for n=1
n = 1
Nn1 = N
M_values_n1 = [1.2040 * N]
lambda_values_n1 = np.array(M_values_n1) / Nn1
blank_term1 = blank / Nn1
K_values_modified_n1 = 1 - np.exp(- (lambda_values_n1 + blank_term1))
plt.scatter(lambda_values_n1, K_values_modified_n1, color='red')
plt.annotate(f'λ={lambda_values_n1[0]:.4f}', (lambda_values_n1[0], K_values_modified_n1[0]),
             textcoords="offset points", xytext=(5, 20), ha='center', fontsize=15, color='red')

# # Annotations for n=1 with very small M
# M_values_n1 = [0.001 * N]
# lambda_values_n1 = np.array(M_values_n1) / Nn1
# blank_term1 = blank / Nn1
# K_values_modified_n1 = 1 - np.exp(- (lambda_values_n1 + blank_term1))
# plt.scatter(lambda_values_n1, K_values_modified_n1, color='red')
# plt.annotate(f'λ={lambda_values_n1[0]:.4f}', (lambda_values_n1[0], K_values_modified_n1[0]),
#              textcoords="offset points", xytext=(5, -10), ha='center', fontsize=15, color='red')

# Annotations for n=13
n = 13
Nn14 = N * left ** (n - 1)
agleft_factor = agleft ** (n - 1)
M_values_n14 = [int(131 * agleft_factor), int(1.2040 * N * agleft_factor)]
lambda_values_n14 = np.array(M_values_n14) / Nn14
blank_term14 = blank / Nn14
K_values_modified_n14 = 1 - np.exp(- (lambda_values_n14 + blank_term14))
plt.scatter(lambda_values_n14, K_values_modified_n14, color='blue')
for i in range(len(lambda_values_n14)):
    plt.annotate(f'λ={lambda_values_n14[i]:.6f}', (lambda_values_n14[i], K_values_modified_n14[i]),
                 textcoords="offset points", xytext=(50, -20), ha='center', fontsize=15, color='blue')

# Chart settings
plt.xlabel('λ (M / (N × left^(n-1)))', fontsize=15)
plt.ylabel('K / (N × left^(n-1))', fontsize=15)
plt.title(f'Relationship between λ and K / (N × left^(n-1))\nInitial N={N}', fontsize=15)
plt.legend(title='Wash times', fontsize=15)
plt.xscale('log')

plt.yscale('log')
# plt.ylim(1e-7, 1)
# Save the plot
dpi_value = 300
plt.savefig('lost_of_ag.png', dpi=dpi_value)
plt.show()

# Save the data to a CSV file
csv_filename = 'wash_data.csv'
header = ['λ (M / (N × left^(n-1)))', 'K / (N × left^(n-1))', 'N', 'wash times']

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(csv_data)  # Write the data

print(f"Data has been saved to {csv_filename}")
