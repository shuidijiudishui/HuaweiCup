import pandas as pd
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

# Step 1: Load Material 1 data (sine wave only)
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# Filter data to only sine wave
sine_wave_data = material_1[material_1.iloc[:, 3] == '正弦波']

# Step 2: Define Steinmetz equation and modified equations with temperature correction
# 在模型函数中添加np.clip以避免溢出
def steinmetz_equation(params, inputs):
    k, a, b = params
    frequency, B_max = inputs
    return k * (np.clip(frequency, 1e-10, 1e10) ** a) * (np.clip(B_max, 1e-10, 1e10) ** b)


# 计算雅可比的函数
def jacobian(params, model, inputs, core_loss):
    predictions = model(params, inputs)
    errors = predictions - core_loss
    jac = np.zeros((len(errors), len(params)))

    for i in range(len(params)):
        temp_params = np.array(params)
        temp_params[i] += 1e-8
        predictions_temp = model(temp_params, inputs)
        jac[:, i] = (predictions_temp - predictions) / 1e-8

    return jac

def modified_steinmetz_linear(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature)

def modified_steinmetz_quadratic(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature + d * temperature**2)

def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

# Step 3: Extract relevant columns: temperature, frequency, B_max, and core loss
temperature = sine_wave_data.iloc[:, 0].values  # Temperature column
frequency = sine_wave_data.iloc[:, 1].values    # Frequency column
core_loss = sine_wave_data.iloc[:, 2].values    # Core loss column
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values  # Peak flux density (B_max)

# Step 4: Define error function for minimize and least_squares
def error_function(params, model, inputs, core_loss):
    return np.sum((model(params, inputs) - core_loss) ** 2)

def error_function_ls(params, model, inputs, core_loss):
    return model(params, inputs) - core_loss

import pandas as pd
import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

# Step 1: Load Material 1 data (sine wave only)
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# Filter data to only sine wave
sine_wave_data = material_1[material_1.iloc[:, 3] == '正弦波']

# Step 2: Define Steinmetz equation and modified equations with temperature correction
def steinmetz_equation(params, inputs):
    k, a, b = params
    frequency, B_max = inputs
    return k * (frequency**a) * (B_max**b)

def modified_steinmetz_linear(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature)

def modified_steinmetz_quadratic(params, inputs):
    k, a, b, c, d = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature + d * temperature**2)

def modified_steinmetz_exponential(params, inputs):
    k, a, b, c = params
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * np.exp(c * temperature)

# Step 3: Extract relevant columns: temperature, frequency, B_max, and core loss
# Step 3: Prepare inputs correctly
inputs_steinmetz = (frequency, B_max)  # 这是一个包含两个数组的元组
inputs_modified = (frequency, B_max, temperature)  # 这是一个包含三个数组的元组

# 输入数组的形状调整
inputs_steinmetz = np.hstack((frequency, B_max))  # 组合为二维数组
inputs_modified = np.hstack((frequency, B_max, temperature))  # 组合为二维数组

# Step 4: Define error function for minimize and least_squares
def error_function(params, model, inputs, core_loss):
    frequency, B_max = inputs  # 解包inputs为frequency和B_max
    return np.sum((model(params, (frequency, B_max)) - core_loss) ** 2)

def error_function_ls(params, model, inputs, core_loss):
    predictions = model(params, inputs)
    return predictions - core_loss  # 返回一维数组

# Step 5: Fit models using BFGS, least_squares, and Newton-CG
# 确保输入的形状一致，尤其在组合输入数组时
inputs_steinmetz = np.hstack((frequency, B_max)).reshape(-1, 2)  # 组合为二维数组
inputs_modified = np.hstack((frequency, B_max, temperature)).reshape(-1, 3)  # 组合为二维数组

# Initial guesses for parameters
initial_params_steinmetz = [1e-5, 1.5, 2.5]
initial_params_linear = [1e-5, 1.5, 2.5, 0.001]
initial_params_quadratic = [1e-5, 1.5, 2.5, 0.001, 0.0001]
initial_params_exponential = [1e-5, 1.5, 2.5, 0.001]

# BFGS Optimization
result_steinmetz_bfgs = minimize(error_function, initial_params_steinmetz, args=(steinmetz_equation, inputs_steinmetz, core_loss), method='BFGS')
result_modified_linear_bfgs = minimize(error_function, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified, core_loss), method='BFGS')
result_modified_quadratic_bfgs = minimize(error_function, initial_params_quadratic, args=(modified_steinmetz_quadratic, inputs_modified, core_loss), method='BFGS')
result_modified_exponential_bfgs = minimize(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified, core_loss), method='BFGS')

# Least Squares Optimization
result_steinmetz_ls = least_squares(error_function_ls, initial_params_steinmetz,
                                     args=(steinmetz_equation, inputs_steinmetz, core_loss))
result_modified_linear_ls = least_squares(error_function_ls, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified, core_loss))
result_modified_quadratic_ls = least_squares(error_function_ls, initial_params_quadratic, args=(modified_steinmetz_quadratic, inputs_modified, core_loss))
result_modified_exponential_ls = least_squares(error_function_ls, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified, core_loss))

# Newton-CG Optimization
# 使用Newton-CG时传入雅可比
result_steinmetz_newton = minimize(error_function, initial_params_steinmetz,
                                   args=(steinmetz_equation, inputs_steinmetz, core_loss),
                                   method='Newton-CG', jac=jacobian)
result_modified_linear_newton = minimize(error_function, initial_params_linear, args=(modified_steinmetz_linear, inputs_modified, core_loss), method='Newton-CG')
result_modified_quadratic_newton = minimize(error_function, initial_params_quadratic, args=(modified_steinmetz_quadratic, inputs_modified, core_loss), method='Newton-CG')
result_modified_exponential_newton = minimize(error_function, initial_params_exponential, args=(modified_steinmetz_exponential, inputs_modified, core_loss), method='Newton-CG')

# Step 6: Evaluate and compare the models
pred_steinmetz_bfgs = steinmetz_equation(result_steinmetz_bfgs.x, inputs_steinmetz)
pred_modified_linear_bfgs = modified_steinmetz_linear(result_modified_linear_bfgs.x, inputs_modified)
pred_modified_quadratic_bfgs = modified_steinmetz_quadratic(result_modified_quadratic_bfgs.x, inputs_modified)
pred_modified_exponential_bfgs = modified_steinmetz_exponential(result_modified_exponential_bfgs.x, inputs_modified)

pred_steinmetz_ls = steinmetz_equation(result_steinmetz_ls.x, inputs_steinmetz)
pred_modified_linear_ls = modified_steinmetz_linear(result_modified_linear_ls.x, inputs_modified)
pred_modified_quadratic_ls = modified_steinmetz_quadratic(result_modified_quadratic_ls.x, inputs_modified)
pred_modified_exponential_ls = modified_steinmetz_exponential(result_modified_exponential_ls.x, inputs_modified)

pred_steinmetz_newton = steinmetz_equation(result_steinmetz_newton.x, inputs_steinmetz)
pred_modified_linear_newton = modified_steinmetz_linear(result_modified_linear_newton.x, inputs_modified)
pred_modified_quadratic_newton = modified_steinmetz_quadratic(result_modified_quadratic_newton.x, inputs_modified)
pred_modified_exponential_newton = modified_steinmetz_exponential(result_modified_exponential_newton.x, inputs_modified)

# Calculate relative errors for both BFGS, least_squares, and Newton-CG
error_steinmetz_bfgs = np.abs(core_loss - pred_steinmetz_bfgs) / core_loss
error_modified_linear_bfgs = np.abs(core_loss - pred_modified_linear_bfgs) / core_loss
error_modified_quadratic_bfgs = np.abs(core_loss - pred_modified_quadratic_bfgs) / core_loss
error_modified_exponential_bfgs = np.abs(core_loss - pred_modified_exponential_bfgs) / core_loss

error_steinmetz_ls = np.abs(core_loss - pred_steinmetz_ls) / core_loss
error_modified_linear_ls = np.abs(core_loss - pred_modified_linear_ls) / core_loss
error_modified_quadratic_ls = np.abs(core_loss - pred_modified_quadratic_ls) / core_loss
error_modified_exponential_ls = np.abs(core_loss - pred_modified_exponential_ls) / core_loss

error_steinmetz_newton = np.abs(core_loss - pred_steinmetz_newton) / core_loss
error_modified_linear_newton = np.abs(core_loss - pred_modified_linear_newton) / core_loss
error_modified_quadratic_newton = np.abs(core_loss - pred_modified_quadratic_newton) / core_loss
error_modified_exponential_newton = np.abs(core_loss - pred_modified_exponential_newton) / core_loss

# Step 7: Plot results for different models (BFGS, Least Squares, Newton-CG)
plt.figure(figsize=(12, 8))

# Steinmetz comparison
plt.subplot(2, 2, 1)
plt.plot(temperature, error_steinmetz_bfgs, label='BFGS Steinmetz Error', marker='o')
plt.plot(temperature, error_steinmetz_ls, label='Least Squares Steinmetz Error', marker='x')
plt.plot(temperature, error_steinmetz_newton, label='Newton-CG Steinmetz Error', marker='s')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Steinmetz Error Comparison')

# Linear Modified Steinmetz comparison
plt.subplot(2, 2, 2)
plt.plot(temperature, error_modified_linear_bfgs, label='BFGS Linear Steinmetz Error', marker='o')
plt.plot(temperature, error_modified_linear_ls, label='Least Squares Linear Steinmetz Error', marker='x')
plt.plot(temperature, error_modified_linear_newton, label='Newton-CG Linear Steinmetz Error', marker='s')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Linear Modified Steinmetz Error Comparison')

# Quadratic Modified Steinmetz comparison
plt.subplot(2, 2, 3)
plt.plot(temperature, error_modified_quadratic_bfgs, label='BFGS Quadratic Steinmetz Error', marker='o')
plt.plot(temperature, error_modified_quadratic_ls, label='Least Squares Quadratic Steinmetz Error', marker='x')
plt.plot(temperature, error_modified_quadratic_newton, label='Newton-CG Quadratic Steinmetz Error', marker='s')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Quadratic Modified Steinmetz Error Comparison')

# Exponential Modified Steinmetz comparison
plt.subplot(2, 2, 4)
plt.plot(temperature, error_modified_exponential_bfgs, label='BFGS Exponential Steinmetz Error', marker='o')
plt.plot(temperature, error_modified_exponential_ls, label='Least Squares Exponential Steinmetz Error', marker='x')
plt.plot(temperature, error_modified_exponential_newton, label='Newton-CG Exponential Steinmetz Error', marker='s')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Exponential Modified Steinmetz Error Comparison')

plt.tight_layout()
plt.show()

# Step 8: Print the average error for each optimization result
print("Average error (Original Steinmetz - BFGS):", np.mean(error_steinmetz_bfgs))
print("Average error (Linear Modified Steinmetz - BFGS):", np.mean(error_modified_linear_bfgs))
print("Average error (Quadratic Modified Steinmetz - BFGS):", np.mean(error_modified_quadratic_bfgs))
print("Average error (Exponential Modified Steinmetz - BFGS):", np.mean(error_modified_exponential_bfgs))

print("Average error (Original Steinmetz - Least Squares):", np.mean(error_steinmetz_ls))
print("Average error (Linear Modified Steinmetz - Least Squares):", np.mean(error_modified_linear_ls))
print("Average error (Quadratic Modified Steinmetz - Least Squares):", np.mean(error_modified_quadratic_ls))
print("Average error (Exponential Modified Steinmetz - Least Squares):", np.mean(error_modified_exponential_ls))

print("Average error (Original Steinmetz - Newton-CG):", np.mean(error_steinmetz_newton))
print("Average error (Linear Modified Steinmetz - Newton-CG):", np.mean(error_modified_linear_newton))
print("Average error (Quadratic Modified Steinmetz - Newton-CG):", np.mean(error_modified_quadratic_newton))
print("Average error (Exponential Modified Steinmetz - Newton-CG):", np.mean(error_modified_exponential_newton))
