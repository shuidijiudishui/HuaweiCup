import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Step 1: Load Material 1 data (sine wave only)
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# Filter data to only sine wave
sine_wave_data = material_1[material_1.iloc[:, 3] == '正弦波']

# Step 2: Define Steinmetz equation with two inputs: frequency and B_max
def steinmetz_equation(inputs, k, a, b):
    frequency, B_max = inputs
    return k * (frequency**a) * (B_max**b)

# Step 3: Define the modified Steinmetz equation with temperature correction
def modified_steinmetz(inputs, k, a, b, c):
    frequency, B_max, temperature = inputs
    return k * (frequency**a) * (B_max**b) * (1 + c * temperature)

# Extract relevant columns: temperature, frequency, B_max (use a representative value from flux density), and core loss
temperature = sine_wave_data.iloc[:, 0].values  # Temperature column
frequency = sine_wave_data.iloc[:, 1].values  # Frequency column
core_loss = sine_wave_data.iloc[:, 2].values  # Core loss column
B_max = sine_wave_data.iloc[:, 4:1029].max(axis=1).values  # Peak flux density (B_max)

# Step 4: Fit original Steinmetz equation (ignoring temperature)
inputs_steinmetz = np.array([frequency, B_max])
popt_steinmetz, _ = curve_fit(steinmetz_equation, inputs_steinmetz, core_loss)

# Step 5: Fit modified Steinmetz equation (with temperature)
inputs_modified = np.array([frequency, B_max, temperature])
popt_modified, _ = curve_fit(modified_steinmetz, inputs_modified, core_loss)

# Step 6: Evaluate and compare the models
pred_steinmetz = steinmetz_equation(inputs_steinmetz, *popt_steinmetz)
pred_modified = modified_steinmetz(inputs_modified, *popt_modified)

# Calculate errors
error_steinmetz = np.abs(core_loss - pred_steinmetz) / core_loss
error_modified = np.abs(core_loss - pred_modified) / core_loss

# Step 7: Plot results
plt.figure(figsize=(10, 6))
plt.plot(temperature, error_steinmetz, label='Original Steinmetz Error', marker='o')
plt.plot(temperature, error_modified, label='Modified Steinmetz Error', marker='x')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Comparison of Steinmetz and Modified Steinmetz Equation Errors')
plt.show()

# Print the average error for both models
print("Average error (Original Steinmetz):", np.mean(error_steinmetz))
print("Average error (Modified Steinmetz):", np.mean(error_modified))