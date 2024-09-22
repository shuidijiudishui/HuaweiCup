import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Step 1: Load the Excel data
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')
material_2 = pd.read_excel(file_path, sheet_name='材料2')
material_3 = pd.read_excel(file_path, sheet_name='材料3')
material_4 = pd.read_excel(file_path, sheet_name='材料4')


# Step 2: Fourier Transform function to extract FFT features
def extract_fft_features(flux_density_data):
    fft_result = fft(flux_density_data)
    # Only keep the first 10 frequency components as features
    features = np.abs(fft_result)[:10]
    return features


# Step 3: Create feature matrix from the materials data
def create_feature_matrix(material_data):
    features = []
    labels = material_data.iloc[:, 3].values  # Column 3 contains waveform types
    for i in range(material_data.shape[0]):
        flux_density = material_data.iloc[i, 4:].values
        fft_features = extract_fft_features(flux_density)
        features.append(fft_features)
    return np.array(features), labels


# Extract features from each material
X1, y1 = create_feature_matrix(material_1)
X2, y2 = create_feature_matrix(material_2)
X3, y3 = create_feature_matrix(material_3)
X4, y4 = create_feature_matrix(material_4)

# Combine all the data
X = np.vstack((X1, X2, X3, X4))
y = np.hstack((y1, y2, y3, y4))


# Step 4: Visualize the FFT features
def visualize_fft_features(X, y):
    waveform_types = ['正弦波', '三角波', '梯形波']
    colors = ['b', 'g', 'r']

    plt.figure(figsize=(12, 6))

    for i, waveform in enumerate(waveform_types):
        # Select samples of the current waveform type
        mask = y == waveform
        features_waveform = X[mask]

        # Calculate mean and std deviation for each FFT feature (first 10)
        mean_features = np.mean(features_waveform, axis=0)
        std_features = np.std(features_waveform, axis=0)

        # Plotting the mean of FFT components with error bars (std dev)
        plt.errorbar(range(1, 11), mean_features, yerr=std_features, fmt='-o', label=f'{waveform}', color=colors[i])

    plt.title('Mean FFT Features for Different Waveforms')
    plt.xlabel('FFT Component')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()


# Visualize the extracted FFT features
visualize_fft_features(X, y)
