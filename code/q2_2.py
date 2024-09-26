import pandas as pd
import numpy as np
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

# Step 1: Load Material 1 data (sine wave only)
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# Filter data to only sine wave
sine_wave_data = material_1[material_1.iloc[:, 3] == '正弦波']

# Step 2: Define Steinmetz equation and modified equations
def steinmetz_equation(params, inputs):
    k, a, b = params
    frequency, B_max = inputs[0], inputs[1]
    return k * (frequency**a) * (B_max**b)

def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

def modified_steinmetz_polynomial(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) + c * temperature + d * temperature**2

# Extract relevant columns
temperature = sine_wave_data.iloc[:, 0].values
frequency = sine_wave_data.iloc[:, 1].values
core_loss = sine_wave_data.iloc[:, 2].values
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values

# Define error function for least_squares
def error_function(params, model, inputs, core_loss):
    return model(params, inputs) - core_loss

# Fit models using least_squares
inputs_steinmetz = np.array([frequency, B_max])
inputs_modified_exponential = np.array([frequency, B_max, temperature])
inputs_modified_polynomial = np.array([frequency, B_max, temperature])

initial_params_steinmetz = [1e-5, 1.5, 2.5]
initial_params_exponential = [1e-5, 1.5, 2.5, 0.001]
initial_params_polynomial = [1e-5, 1.5, 2.5, 0.001, 0.001]

result_steinmetz_ls = least_squares(error_function, initial_params_steinmetz, args=(steinmetz_equation, inputs_steinmetz, core_loss))
result_modified_exponential_ls = least_squares(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified_exponential, core_loss))
result_modified_polynomial_ls = least_squares(error_function, initial_params_polynomial, args=(modified_steinmetz_polynomial, inputs_modified_polynomial, core_loss))

# Calculate predictions
pred_steinmetz_ls = steinmetz_equation(result_steinmetz_ls.x, inputs_steinmetz)
pred_modified_exponential_ls = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_modified_exponential)
pred_modified_polynomial_ls = modified_steinmetz_polynomial(result_modified_polynomial_ls.x, inputs_modified_polynomial)

# Calculate errors
error_steinmetz_ls = np.abs(core_loss - pred_steinmetz_ls) / core_loss
error_modified_exponential_ls = np.abs(core_loss - pred_modified_exponential_ls) / core_loss
error_modified_polynomial_ls = np.abs(core_loss - pred_modified_polynomial_ls) / core_loss

# Plot results for different models
plt.figure(figsize=(10, 6))
plt.plot(temperature, error_steinmetz_ls, label='Original Steinmetz Error (LS)', marker='o')
plt.plot(temperature, error_modified_exponential_ls, label='Exponential Modified Steinmetz Error (LS)', marker='s')
plt.plot(temperature, error_modified_polynomial_ls, label='Polynomial Modified Steinmetz Error (LS)', marker='*')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Comparison of Steinmetz and Modified Steinmetz Equation Errors')
plt.show()

# Average error for all models
print("Average error (Original Steinmetz LS):", np.mean(error_steinmetz_ls))
print("Average error (Exponential Modified Steinmetz LS):", np.mean(error_modified_exponential_ls))
print("Average error (Polynomial Modified Steinmetz LS):", np.mean(error_modified_polynomial_ls))

# Scatter plots for different temperatures
temperatures = [25, 50, 70, 90]
for temp in temperatures:
    temp_data = sine_wave_data[sine_wave_data.iloc[:, 0] == temp]
    frequency_temp = temp_data.iloc[:, 1].values
    core_loss_temp = temp_data.iloc[:, 2].values
    B_max_temp = temp_data.iloc[:, 4:1029].max(axis=1).values

    inputs_temp_steinmetz = np.array([frequency_temp, B_max_temp])
    inputs_temp_exponential = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])
    inputs_temp_polynomial = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])

    pred_steinmetz_ls_temp = steinmetz_equation(result_steinmetz_ls.x, inputs_temp_steinmetz)
    pred_modified_exponential_ls_temp = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_temp_exponential)
    pred_modified_polynomial_ls_temp = modified_steinmetz_polynomial(result_modified_polynomial_ls.x, inputs_temp_polynomial)

    plt.figure(figsize=(8, 6))
    plt.scatter(frequency_temp, core_loss_temp, label='Measured Core Loss', color='blue', s=20)
    plt.plot(frequency_temp, pred_steinmetz_ls_temp, label='Original Steinmetz (LS)', linestyle='-', color='green')
    plt.plot(frequency_temp, pred_modified_exponential_ls_temp, label='Exponential Modified Steinmetz (LS)', linestyle=':', color='orange')
    plt.plot(frequency_temp, pred_modified_polynomial_ls_temp, label='Polynomial Modified Steinmetz (LS)', linestyle='--', color='red')
    plt.title(f'Core Loss vs Frequency at {temp}°C')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Core Loss (W/m^3)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Histogram of errors for each model
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.hist(error_steinmetz_ls, bins=30, color='blue', alpha=0.7)
plt.title('Original Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(error_modified_exponential_ls, bins=30, color='orange', alpha=0.7)
plt.title('Exponential Modified Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(error_modified_polynomial_ls, bins=30, color='red', alpha=0.7)
plt.title('Polynomial Modified Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
