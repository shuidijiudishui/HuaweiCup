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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Function to calculate and print evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"Evaluation for {model_name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}\n")
    return r2, mae, mse
def steinmetz_equation(params, inputs):
    k, a, b = params
    frequency, B_max = inputs[0], inputs[1]  # Extract frequency and B_max from inputs
    return k * (frequency**a) * (B_max**b)

# Define modified Steinmetz equations with temperature correction
def modified_steinmetz_linear(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature)

def modified_steinmetz_quadratic(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature + d * temperature**2)

def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

def modified_steinmetz_logarithmic(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs[0], inputs[1], inputs[2]
    return k * (frequency**a) * (B_max**b) * np.log(temperature + c)

# Extract relevant columns: temperature, frequency, B_max, and core loss
temperature = sine_wave_data.iloc[:, 0].values  # Temperature column
frequency = sine_wave_data.iloc[:, 1].values  # Frequency column
core_loss = sine_wave_data.iloc[:, 2].values  # Core loss column
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values  # Peak flux density (B_max)

# Define error function for least_squares
def error_function(params, model, inputs, core_loss):
    return model(params, inputs) - core_loss

# Step 4: Fit models using least_squares and L-BFGS-B
inputs_steinmetz = np.array([frequency, B_max])
inputs_modified = np.array([frequency, B_max, temperature])

# Initial guesses for parameters (can be adjusted)
initial_params_steinmetz = [1e-5, 1.5, 2.5]
initial_params_linear = [1e-5, 1.5, 2.5, 0.001]
initial_params_quadratic = [1e-5, 1.5, 2.5, 0.001, 0.0001]
initial_params_exponential = [1e-5, 1.5, 2.5, 0.001]
initial_params_logarithmic = [1e-5, 1.5, 2.5, 1.0]  # Added for logarithmic model

# Fitting original Steinmetz equation using least_squares
result_steinmetz_ls = least_squares(error_function, initial_params_steinmetz, args=(steinmetz_equation, inputs_steinmetz, core_loss))

# Fitting modified Steinmetz equations using least_squares
result_modified_linear_ls = least_squares(error_function, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified, core_loss))
result_modified_quadratic_ls = least_squares(error_function, initial_params_quadratic, args=(modified_steinmetz_quadratic, inputs_modified, core_loss))
result_modified_exponential_ls = least_squares(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified, core_loss))
result_modified_logarithmic_ls = least_squares(error_function, initial_params_logarithmic, args=(modified_steinmetz_logarithmic, inputs_modified, core_loss))

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

result_modified_logarithmic_lbfgsb = minimize(lambda params: np.sum(error_function(params, modified_steinmetz_logarithmic, inputs_modified, core_loss)**2),
                                               initial_params_logarithmic,
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

pred_modified_logarithmic_ls = modified_steinmetz_logarithmic(result_modified_logarithmic_ls.x, inputs_modified)
pred_modified_logarithmic_lbfgsb = modified_steinmetz_logarithmic(result_modified_logarithmic_lbfgsb.x, inputs_modified)

# Calculate errors
error_steinmetz_ls = np.abs(core_loss - pred_steinmetz_ls) / core_loss
error_steinmetz_lbfgsb = np.abs(core_loss - pred_steinmetz_lbfgsb) / core_loss

error_modified_linear_ls = np.abs(core_loss - pred_modified_linear_ls) / core_loss
error_modified_linear_lbfgsb = np.abs(core_loss - pred_modified_linear_lbfgsb) / core_loss

error_modified_quadratic_ls = np.abs(core_loss - pred_modified_quadratic_ls) / core_loss
error_modified_quadratic_lbfgsb = np.abs(core_loss - pred_modified_quadratic_lbfgsb) / core_loss

error_modified_exponential_ls = np.abs(core_loss - pred_modified_exponential_ls) / core_loss
error_modified_exponential_lbfgsb = np.abs(core_loss - pred_modified_exponential_lbfgsb) / core_loss

error_modified_logarithmic_ls = np.abs(core_loss - pred_modified_logarithmic_ls) / core_loss
error_modified_logarithmic_lbfgsb = np.abs(core_loss - pred_modified_logarithmic_lbfgsb) / core_loss

# Step 6: Plot results for different models including L-BFGS-B
# 设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

plt.figure(figsize=(10, 6))
plt.plot(temperature, error_steinmetz_ls, label='Original Steinmetz Error (LS)', marker='o')
plt.plot(temperature, error_steinmetz_lbfgsb, label='Original Steinmetz Error (L-BFGS-B)', marker='*')
plt.plot(temperature, error_modified_linear_ls, label='Linear Modified Steinmetz Error (LS)', marker='x')
plt.plot(temperature, error_modified_linear_lbfgsb, label='Linear Modified Steinmetz Error (L-BFGS-B)', marker='x', linestyle='--')
plt.plot(temperature, error_modified_quadratic_ls, label='Quadratic Modified Steinmetz Error (LS)', marker='^')
plt.plot(temperature, error_modified_quadratic_lbfgsb, label='Quadratic Modified Steinmetz Error (L-BFGS-B)', marker='^', linestyle='--')
plt.plot(temperature, error_modified_exponential_ls, label='Exponential Modified Steinmetz Error (LS)', marker='s')
plt.plot(temperature, error_modified_exponential_lbfgsb, label='Exponential Modified Steinmetz Error (L-BFGS-B)', marker='s', linestyle='--')
plt.plot(temperature, error_modified_logarithmic_ls, label='Logarithmic Modified Steinmetz Error (LS)', marker='D')
plt.plot(temperature, error_modified_logarithmic_lbfgsb, label='Logarithmic Modified Steinmetz Error (L-BFGS-B)', marker='D', linestyle='--')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.title('Model Comparison: Error Analysis')
plt.legend()
plt.grid()
plt.show()

# Step 6: Evaluate all models and print errors
models = {
    'Original Steinmetz LS': (error_steinmetz_ls, result_steinmetz_ls, steinmetz_equation),
    'Original Steinmetz L-BFGS-B': (error_steinmetz_lbfgsb, result_steinmetz_lbfgsb, steinmetz_equation),
    'Linear Modified Steinmetz LS': (error_modified_linear_ls, result_modified_linear_ls, modified_steinmetz_linear),
    'Linear Modified Steinmetz L-BFGS-B': (
    error_modified_linear_lbfgsb, result_modified_linear_lbfgsb, modified_steinmetz_linear),
    'Quadratic Modified Steinmetz LS': (
    error_modified_quadratic_ls, result_modified_quadratic_ls, modified_steinmetz_quadratic),
    'Quadratic Modified Steinmetz L-BFGS-B': (
    error_modified_quadratic_lbfgsb, result_modified_quadratic_lbfgsb, modified_steinmetz_quadratic),
    'Exponential Modified Steinmetz LS': (
    error_modified_exponential_ls, result_modified_exponential_ls, modified_steinmetz_exponential),
    'Exponential Modified Steinmetz L-BFGS-B': (
    error_modified_exponential_lbfgsb, result_modified_exponential_lbfgsb, modified_steinmetz_exponential),
    'Logarithmic Modified Steinmetz LS': (
    error_modified_logarithmic_ls, result_modified_logarithmic_ls, modified_steinmetz_logarithmic),
    'Logarithmic Modified Steinmetz L-BFGS-B': (
    error_modified_logarithmic_lbfgsb, result_modified_logarithmic_lbfgsb, modified_steinmetz_logarithmic),
}

# Prepare to collect metrics
metrics = {
    'Model': [],
    'MAE': [],
    'MSE': [],
    'R²': []
}

# Iterate over the models to calculate evaluation metrics
for model_name, (error, result, model_func) in models.items():
    # Actual core loss values
    y_true = sine_wave_data.iloc[:, 2].values  # Actual core loss
    # Predicted values using the model function
    inputs = np.array([sine_wave_data.iloc[:, 1].values, sine_wave_data.iloc[:, 4:1029].max(axis=1).values])
    if 'Modified' in model_name:
        inputs = np.array([sine_wave_data.iloc[:, 1].values, sine_wave_data.iloc[:, 4:1029].max(axis=1).values,
                           sine_wave_data.iloc[:, 0].values])  # Include temperature for modified models

    y_pred = model_func(result.x, inputs)  # Predicted core loss

    # Evaluate the model
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Store metrics
    metrics['Model'].append(model_name)
    metrics['MAE'].append(mae)
    metrics['MSE'].append(mse)
    metrics['R²'].append(r2)
    print(metrics['MAE'])

# Convert metrics to a DataFrame for easier plotting
import pandas as pd
metrics_df = pd.DataFrame(metrics)

# Plot histograms for MAE, MSE, and R²
plt.figure(figsize=(18, 12))

# Plot MAE
plt.subplot(1, 3, 1)
plt.bar(metrics_df['Model'], metrics_df['MAE'], color='blue', alpha=0.7)
plt.title('MAE for Different Models')
plt.ylabel('MAE')
# plt.xlabel('', fontsize=8)
plt.xticks(rotation=45)

# Plot MSE
plt.subplot(1, 3, 2)
plt.bar(metrics_df['Model'], metrics_df['MSE'], color='green', alpha=0.7)
plt.title('MSE for Different Models')
plt.ylabel('MSE')
# plt.xlabel('', fontsize=8)
plt.xticks(rotation=45)

# Plot R²
plt.subplot(1, 3, 3)
plt.bar(metrics_df['Model'], metrics_df['R²'], color='red', alpha=0.7)
plt.title('R² for Different Models')
plt.ylabel('R²')
# plt.xlabel('', fontsize=8)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print the average error for all models including the new correction factor
print("Average error (Original Steinmetz LS):", np.mean(error_steinmetz_ls))
print("Average error (Original Steinmetz L-BFGS-B):", np.mean(error_steinmetz_lbfgsb))
print("Average error (Linear Modified Steinmetz LS):", np.mean(error_modified_linear_ls))
print("Average error (Linear Modified Steinmetz L-BFGS-B):", np.mean(error_modified_linear_lbfgsb))
print("Average error (Quadratic Modified Steinmetz LS):", np.mean(error_modified_quadratic_ls))
print("Average error (Quadratic Modified Steinmetz L-BFGS-B):", np.mean(error_modified_quadratic_lbfgsb))
print("Average error (Exponential Modified Steinmetz LS):", np.mean(error_modified_exponential_ls))
print("Average error (Exponential Modified Steinmetz L-BFGS-B):", np.mean(error_modified_exponential_lbfgsb))

# New correction factor: k * (frequency**a) * (B_max**b) * np.log(temperature + c)
print("Average error (Logarithmic Modified Steinmetz LS):", np.mean(error_modified_logarithmic_ls))
print("Average error (Logarithmic Modified Steinmetz L-BFGS-B):", np.mean(error_modified_logarithmic_lbfgsb))

# Step 7: Plot scatter plots of core loss vs frequency for different temperatures, with fitted curves
temperatures = [25, 50, 70, 90]  # The four different temperature values

# Scatter plot and fitted curves for each temperature
for temp in temperatures:
    # Filter data for the specific temperature
    temp_data = sine_wave_data[sine_wave_data.iloc[:, 0] == temp]

    # Extract corresponding columns: frequency, core loss, B_max
    frequency_temp = temp_data.iloc[:, 1].values
    core_loss_temp = temp_data.iloc[:, 2].values
    B_max_temp = temp_data.iloc[:, 4:1029].max(axis=1).values  # Peak flux density (B_max)

    # Prepare inputs for the models
    inputs_temp_steinmetz = np.array([frequency_temp, B_max_temp])
    inputs_temp_modified = np.array(
        [frequency_temp, B_max_temp, np.full_like(frequency_temp, temp)])  # Temp is constant here

    # Predictions for each model at this temperature
    pred_steinmetz_ls_temp = steinmetz_equation(result_steinmetz_ls.x, inputs_temp_steinmetz)
    pred_modified_linear_ls_temp = modified_steinmetz_linear(result_modified_linear_ls.x, inputs_temp_modified)
    pred_modified_quadratic_ls_temp = modified_steinmetz_quadratic(result_modified_quadratic_ls.x, inputs_temp_modified)
    pred_modified_exponential_ls_temp = modified_steinmetz_exponential(result_modified_exponential_ls.x,
                                                                       inputs_temp_modified)
    pred_modified_logarithmic_ls_temp = modified_steinmetz_logarithmic(result_modified_logarithmic_ls.x, inputs_temp_modified)

    # Plot scatter plot and fitted curves
    plt.figure(figsize=(8, 6))

    # Scatter plot of the original data
    plt.scatter(frequency_temp, core_loss_temp, label='Measured Core Loss', color='blue', s=20)

    # Fitted curves from different models
    plt.plot(frequency_temp, pred_steinmetz_ls_temp, label='Original Steinmetz (LS)', linestyle='-', color='green')
    plt.plot(frequency_temp, pred_modified_linear_ls_temp, label='Linear Modified Steinmetz (LS)', linestyle='--', color='red')
    plt.plot(frequency_temp, pred_modified_quadratic_ls_temp, label='Quadratic Modified Steinmetz (LS)', linestyle='-.', color='purple')
    plt.plot(frequency_temp, pred_modified_exponential_ls_temp, label='Exponential Modified Steinmetz (LS)', linestyle=':', color='orange')
    plt.plot(frequency_temp, pred_modified_logarithmic_ls_temp, label='Logarithmic Modified Steinmetz (LS)', linestyle='-', color='black')

    # Plot labels and title
    plt.title(f'Core Loss vs Frequency at {temp}°C')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Core Loss (W/m^3)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Step 8: Plot histogram of errors for each model
plt.figure(figsize=(16, 12))

# Plot histogram for original Steinmetz error (LS)
plt.subplot(3, 3, 1)
plt.hist(error_steinmetz_ls, bins=30, color='blue', alpha=0.7)
plt.title('Original Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for original Steinmetz error (L-BFGS-B)
plt.subplot(3, 3, 2)
plt.hist(error_steinmetz_lbfgsb, bins=30, color='green', alpha=0.7)
plt.title('Original Steinmetz Error (L-BFGS-B)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for linear modified Steinmetz error (LS)
plt.subplot(3, 3, 3)
plt.hist(error_modified_linear_ls, bins=30, color='red', alpha=0.7)
plt.title('Linear Modified Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for linear modified Steinmetz error (L-BFGS-B)
plt.subplot(3, 3, 4)
plt.hist(error_modified_linear_lbfgsb, bins=30, color='purple', alpha=0.7)
plt.title('Linear Modified Steinmetz Error (L-BFGS-B)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for quadratic modified Steinmetz error (LS)
plt.subplot(3, 3, 5)
plt.hist(error_modified_quadratic_ls, bins=30, color='orange', alpha=0.7)
plt.title('Quadratic Modified Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for quadratic modified Steinmetz error (L-BFGS-B)
plt.subplot(3, 3, 6)
plt.hist(error_modified_quadratic_lbfgsb, bins=30, color='brown', alpha=0.7)
plt.title('Quadratic Modified Steinmetz Error (L-BFGS-B)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for logarithmic modified Steinmetz error (LS)
plt.subplot(3, 3, 7)
plt.hist(error_modified_logarithmic_ls, bins=30, color='black', alpha=0.7)
plt.title('Logarithmic Modified Steinmetz Error (LS)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

# Plot histogram for logarithmic modified Steinmetz error (L-BFGS-B)
plt.subplot(3, 3, 8)
plt.hist(error_modified_logarithmic_lbfgsb, bins=30, color='gray', alpha=0.7)
plt.title('Logarithmic Modified Steinmetz Error (L-BFGS-B)')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
