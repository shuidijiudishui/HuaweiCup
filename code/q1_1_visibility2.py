import pandas as pd
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


# Step 1: Define Fourier Transform function to extract FFT features
def extract_fft_features(flux_density_data):
    fft_result = fft(flux_density_data)
    # Only keep the first 10 frequency components as features
    features = np.abs(fft_result)[:10]
    return features


# Step 2: Visualize FFT feature proportions using pie charts
def plot_fft_feature_pie(fft_features, title):
    total = np.sum(fft_features)
    proportions = fft_features / total
    labels = [f'Component {i + 1}' for i in range(len(fft_features))]

    plt.figure(figsize=(6, 6))
    plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()


# Example data (replace these with actual data from '附件一')
# Simulating flux density data for different waveforms
# Replace the following lists with real flux density data for each waveform
sine_wave_data = np.sin(np.linspace(0, 2 * np.pi, 1024))  # Simulated sine wave data
triangle_wave_data = np.abs(np.mod(np.linspace(0, 4, 1024), 2) - 1)  # Simulated triangle wave
trapezoid_wave_data = np.clip(np.sin(np.linspace(0, 2 * np.pi, 1024)) * 2, -1, 1)  # Simulated trapezoid wave

# Step 3: Extract FFT features for each waveform type
fft_sine = extract_fft_features(sine_wave_data)
fft_triangle = extract_fft_features(triangle_wave_data)
fft_trapezoid = extract_fft_features(trapezoid_wave_data)

# Step 4: Plot pie charts for each waveform's FFT features
plot_fft_feature_pie(fft_sine, 'Sine Wave FFT Feature Proportions')
plot_fft_feature_pie(fft_triangle, 'Triangle Wave FFT Feature Proportions')
plot_fft_feature_pie(fft_trapezoid, 'Trapezoid Wave FFT Feature Proportions')
