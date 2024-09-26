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
    frequency, B_max = inputs[0], inputs[1]
    return k * (frequency**a) * (B_max**b)

# Define modified Steinmetz equations with temperature correction
def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

def modified_steinmetz_polynomial(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) + c * temperature + d * temperature**2

def modified_steinmetz_logarithmic(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.log(temperature + c)

def modified_steinmetz_linear(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) + c * temperature

# Extract relevant columns
temperature = sine_wave_data.iloc[:, 0].values
frequency = sine_wave_data.iloc[:, 1].values
core_loss = sine_wave_data.iloc[:, 2].values
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values

# Define error function for least_squares
def error_function(params, model, inputs, core_loss):
    return model(params, inputs) - core_loss

# Step 4: Fit models using least_squares and L-BFGS-B
inputs_steinmetz = np.array([frequency, B_max])
inputs_modified_exponential = np.array([frequency, B_max, temperature])
inputs_modified_polynomial = np.array([frequency, B_max, temperature])
inputs_modified_logarithmic = np.array([frequency, B_max, temperature])
inputs_modified_linear = np.array([frequency, B_max, temperature])

# Initial guesses for parameters
initial_params_steinmetz = [1e-5, 1.5, 2.5]
initial_params_exponential = [1e-5, 1.5, 2.5, 0.001]
initial_params_polynomial = [1e-5, 1.5, 2.5, 0.001, 0.001]
initial_params_logarithmic = [1e-5, 1.5, 2.5, 0.1]
initial_params_linear = [1e-5, 1.5, 2.5, 0.1]

# Fitting models using least_squares
result_steinmetz_ls = least_squares(error_function, initial_params_steinmetz, args=(steinmetz_equation, inputs_steinmetz, core_loss))
result_modified_exponential_ls = least_squares(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified_exponential, core_loss))
result_modified_polynomial_ls = least_squares(error_function, initial_params_polynomial, args=(modified_steinmetz_polynomial, inputs_modified_polynomial, core_loss))
result_modified_logarithmic_ls = least_squares(error_function, initial_params_logarithmic, args=(modified_steinmetz_logarithmic, inputs_modified_logarithmic, core_loss))
result_modified_linear_ls = least_squares(error_function, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified_linear, core_loss))

# Fitting using L-BFGS-B for all models
result_steinmetz_lbfgsb = minimize(lambda params: np.sum(error_function(params, steinmetz_equation, inputs_steinmetz, core_loss)**2), initial_params_steinmetz, method='L-BFGS-B')
result_modified_exponential_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_exponential, inputs_modified_exponential, core_loss)**2), initial_params_exponential, method='L-BFGS-B')
result_modified_polynomial_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_polynomial, inputs_modified_polynomial, core_loss)**2), initial_params_polynomial, method='L-BFGS-B')
result_modified_logarithmic_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_logarithmic, inputs_modified_logarithmic, core_loss)**2), initial_params_logarithmic, method='L-BFGS-B')
result_modified_linear_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_linear, inputs_modified_linear, core_loss)**2), initial_params_linear, method='L-BFGS-B')

# Step 5: Evaluate and compare the models
pred_steinmetz_ls = steinmetz_equation(result_steinmetz_ls.x, inputs_steinmetz)
pred_modified_exponential_ls = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_modified_exponential)
pred_modified_polynomial_ls = modified_steinmetz_polynomial(result_modified_polynomial_ls.x, inputs_modified_polynomial)
pred_modified_logarithmic_ls = modified_steinmetz_logarithmic(result_modified_logarithmic_ls.x, inputs_modified_logarithmic)
pred_modified_linear_ls = modified_steinmetz_linear(result_modified_linear_ls.x, inputs_modified_linear)

# Calculate errors
error_steinmetz_ls = np.abs(core_loss - pred_steinmetz_ls) / core_loss
error_modified_exponential_ls = np.abs(core_loss - pred_modified_exponential_ls) / core_loss
error_modified_polynomial_ls = np.abs(core_loss - pred_modified_polynomial_ls) / core_loss
error_modified_logarithmic_ls = np.abs(core_loss - pred_modified_logarithmic_ls) / core_loss
error_modified_linear_ls = np.abs(core_loss - pred_modified_linear_ls) / core_loss

# Step 6: Plot results for different models
plt.figure(figsize=(10, 6))
plt.plot(temperature, error_steinmetz_ls, label='Original Steinmetz Error (LS)', marker='o')
plt.plot(temperature, error_modified_exponential_ls, label='Exponential Modified Steinmetz Error (LS)', marker='s')
plt.plot(temperature, error_modified_polynomial_ls, label='Polynomial Modified Steinmetz Error (LS)', marker='*')
plt.plot(temperature, error_modified_logarithmic_ls, label='Logarithmic Modified Steinmetz Error (LS)', marker='x')
plt.plot(temperature, error_modified_linear_ls, label='Linear Modified Steinmetz Error (LS)', marker='D')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Comparison of Steinmetz and Modified Steinmetz Equation Errors')
plt.grid()
plt.show()

# Step 7: Plot scatter plots of core loss vs frequency for different temperatures, with fitted curves
temperatures = [25, 50, 70, 90]

# Scatter plot and fitted curves for each temperature
for temp in temperatures:
    # Filter data for the specific temperature
    temp_data = sine_wave_data[sine_wave_data.iloc[:, 0] == temp]

    # Extract corresponding columns
    frequency_temp = temp_data.iloc[:, 1].values
    core_loss_temp = temp_data.iloc[:, 2].values
    B_max_temp = temp_data.iloc[:, 4:1029].max(axis=1).values

    # Prepare inputs for the models
    inputs_temp_steinmetz = np.array([frequency_temp, B_max_temp])
    inputs_temp_exponential = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])
    inputs_temp_polynomial = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])
    inputs_temp_logarithmic = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])
    inputs_temp_linear = np.array([frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])

    # Predictions for each model at this temperature
    pred_steinmetz_ls_temp = steinmetz_equation(result_steinmetz_ls.x, inputs_temp_steinmetz)
    pred_modified_exponential_ls_temp = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_temp_exponential)
    pred_modified_polynomial_ls_temp = modified_steinmetz_polynomial(result_modified_polynomial_ls.x, inputs_temp_polynomial)
    pred_modified_logarithmic_ls_temp = modified_steinmetz_logarithmic(result_modified_logarithmic_ls.x, inputs_temp_logarithmic)
    pred_modified_linear_ls_temp = modified_steinmetz_linear(result_modified_linear_ls.x, inputs_temp_linear)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(frequency_temp, core_loss_temp, label='Core Loss Data', color='black', s=10)
    plt.plot(frequency_temp, pred_steinmetz_ls_temp, label='Steinmetz Fit', color='red')
    plt.plot(frequency_temp, pred_modified_exponential_ls_temp, label='Exponential Modified Fit', color='green')
    plt.plot(frequency_temp, pred_modified_polynomial_ls_temp, label='Polynomial Modified Fit', color='blue')
    plt.plot(frequency_temp, pred_modified_logarithmic_ls_temp, label='Logarithmic Modified Fit', color='purple')
    plt.plot(frequency_temp, pred_modified_linear_ls_temp, label='Linear Modified Fit', color='orange')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Core Loss (W/m³)')
    plt.title(f'Core Loss vs Frequency at {temp} °C')
    plt.legend()
    plt.grid()
    plt.show()
