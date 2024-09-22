import pandas as pd
import numpy as np
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

# Step 1: Load Material 1 data (sine wave only)
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# Filter data to only sine wave
sine_wave_data = material_1[material_1.iloc[:, 3] == '正弦波']

# Step 2: Define Steinmetz equation with two inputs: frequency and B_max
def steinmetz_equation(params, inputs):
    k, a, b = params
    frequency, B_max = inputs[0], inputs[1]  # Extract frequency and B_max from inputs
    return k * (frequency**a) * (B_max**b)

# Define different modified Steinmetz equations with temperature correction
def modified_steinmetz_linear(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]  # Adjusted for inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature)

def modified_steinmetz_quadratic(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature + d * temperature**2)

def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

# Extract relevant columns: temperature, frequency, B_max, and core loss
temperature = sine_wave_data.iloc[:, 0].values  # Temperature column
frequency = sine_wave_data.iloc[:, 1].values  # Frequency column
core_loss = sine_wave_data.iloc[:, 2].values  # Core loss column
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values  # Peak flux density (B_max)

# Define error function for least_squares
def error_function(params, model, inputs, core_loss):
    return model(params, inputs) - core_loss

# Step 4: Fit models using least_squares and L-BFGS-B
inputs_steinmetz = np.array([frequency, B_max])  # Corrected
inputs_modified = np.array([frequency, B_max, temperature])  # Corrected

# Initial guesses for parameters (can be adjusted)
initial_params_steinmetz = [1e-5, 1.5, 2.5]
initial_params_linear = [1e-5, 1.5, 2.5, 0.001]
initial_params_quadratic = [1e-5, 1.5, 2.5, 0.001, 0.0001]
initial_params_exponential = [1e-5, 1.5, 2.5, 0.001]

# Fitting original Steinmetz equation using least_squares
result_steinmetz_ls = least_squares(error_function, initial_params_steinmetz, args=(steinmetz_equation, inputs_steinmetz, core_loss))

# Fitting modified Steinmetz equations using least_squares
result_modified_linear_ls = least_squares(error_function, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified, core_loss))
result_modified_quadratic_ls = least_squares(error_function, initial_params_quadratic, args=(modified_steinmetz_quadratic, inputs_modified, core_loss))
result_modified_exponential_ls = least_squares(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified, core_loss))

# Fitting using L-BFGS-B for the original Steinmetz equation
result_steinmetz_lbfgsb = minimize(lambda params: np.sum(error_function(params, steinmetz_equation, inputs_steinmetz, core_loss)**2),
                                    initial_params_steinmetz,
                                    method='L-BFGS-B')

# Fitting using L-BFGS-B for modified Steinmetz equations
result_modified_linear_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_linear, inputs_modified, core_loss)**2),
                                          initial_params_linear,
                                          method='L-BFGS-B')

result_modified_quadratic_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_quadratic, inputs_modified, core_loss)**2),
                                             initial_params_quadratic,
                                             method='L-BFGS-B')

result_modified_exponential_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_exponential, inputs_modified, core_loss)**2),
                                               initial_params_exponential,
                                               method='L-BFGS-B')

# Step 5: Evaluate and compare the models
pred_steinmetz_ls = steinmetz_equation(result_steinmetz_ls.x, inputs_steinmetz)
pred_steinmetz_lbfgsb = steinmetz_equation(result_steinmetz_lbfgsb.x, inputs_steinmetz)

pred_modified_linear_ls = modified_steinmetz_linear(result_modified_linear_ls.x, inputs_modified)
pred_modified_linear_lbfgsb = modified_steinmetz_linear(result_modified_linear_lbfgsb.x, inputs_modified)

pred_modified_quadratic_ls = modified_steinmetz_quadratic(result_modified_quadratic_ls.x, inputs_modified)
pred_modified_quadratic_lbfgsb = modified_steinmetz_quadratic(result_modified_quadratic_lbfgsb.x, inputs_modified)

pred_modified_exponential_ls = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_modified)
pred_modified_exponential_lbfgsb = modified_steinmetz_exponential(result_modified_exponential_lbfgsb.x, inputs_modified)

# Calculate errors
error_steinmetz_ls = np.abs(core_loss - pred_steinmetz_ls) / core_loss
error_steinmetz_lbfgsb = np.abs(core_loss - pred_steinmetz_lbfgsb) / core_loss

error_modified_linear_ls = np.abs(core_loss - pred_modified_linear_ls) / core_loss
error_modified_linear_lbfgsb = np.abs(core_loss - pred_modified_linear_lbfgsb) / core_loss

error_modified_quadratic_ls = np.abs(core_loss - pred_modified_quadratic_ls) / core_loss
error_modified_quadratic_lbfgsb = np.abs(core_loss - pred_modified_quadratic_lbfgsb) / core_loss

error_modified_exponential_ls = np.abs(core_loss - pred_modified_exponential_ls) / core_loss
error_modified_exponential_lbfgsb = np.abs(core_loss - pred_modified_exponential_lbfgsb) / core_loss

# Step 6: Plot results for different models including L-BFGS-B
plt.figure(figsize=(10, 6))
plt.plot(temperature, error_steinmetz_ls, label='Original Steinmetz Error (LS)', marker='o')
plt.plot(temperature, error_steinmetz_lbfgsb, label='Original Steinmetz Error (L-BFGS-B)', marker='*')
plt.plot(temperature, error_modified_linear_ls, label='Linear Modified Steinmetz Error (LS)', marker='x')
plt.plot(temperature, error_modified_linear_lbfgsb, label='Linear Modified Steinmetz Error (L-BFGS-B)', marker='x', linestyle='--')
plt.plot(temperature, error_modified_quadratic_ls, label='Quadratic Modified Steinmetz Error (LS)', marker='^')
plt.plot(temperature, error_modified_quadratic_lbfgsb, label='Quadratic Modified Steinmetz Error (L-BFGS-B)', marker='^', linestyle='--')
plt.plot(temperature, error_modified_exponential_ls, label='Exponential Modified Steinmetz Error (LS)', marker='s')
plt.plot(temperature, error_modified_exponential_lbfgsb, label='Exponential Modified Steinmetz Error (L-BFGS-B)', marker='s', linestyle='--')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Comparison of Steinmetz and Modified Steinmetz Equation Errors')
plt.show()

# Print the average error for all models including L-BFGS-B
print("Average error (Original Steinmetz LS):", np.mean(error_steinmetz_ls))
print("Average error (Original Steinmetz L-BFGS-B):", np.mean(error_steinmetz_lbfgsb))
print("Average error (Linear Modified Steinmetz LS):", np.mean(error_modified_linear_ls))
print("Average error (Linear Modified Steinmetz L-BFGS-B):", np.mean(error_modified_linear_lbfgsb))
print("Average error (Quadratic Modified Steinmetz LS):", np.mean(error_modified_quadratic_ls))
print("Average error (Quadratic Modified Steinmetz L-BFGS-B):", np.mean(error_modified_quadratic_lbfgsb))
print("Average error (Exponential Modified Steinmetz LS):", np.mean(error_modified_exponential_ls))
print("Average error (Exponential Modified Steinmetz L-BFGS-B):", np.mean(error_modified_exponential_lbfgsb))
