import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load the Excel data
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')
material_2 = pd.read_excel(file_path, sheet_name='材料2')
material_3 = pd.read_excel(file_path, sheet_name='材料3')
material_4 = pd.read_excel(file_path, sheet_name='材料4')

# Step 2: Fourier Transform function to extract FFT features
def extract_fft_features(flux_density_data):
    # Convert the flux density data to float (ignoring any non-numeric data)
    flux_density_data = flux_density_data.astype(float)
    fft_result = fft(flux_density_data)
    features = np.abs(fft_result)[:10]  # Extract first 10 FFT features
    return features

# Step 3: Create feature matrix from the materials data
def create_feature_matrix(material_data):
    features = []
    labels = material_data.iloc[:, 3].values  # Column 3 contains waveform types
    for i in range(material_data.shape[0]):
        flux_density = material_data.iloc[i, 4:].values  # Extract flux density data from column 5 onwards
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

# Step 4: Build and train the classification model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the training dataset
y_pred_train = clf.predict(X_test)
print(classification_report(y_test, y_pred_train))

# Step 5: Predict waveform types in 附件二（测试集）
test_data_2 = pd.read_excel('附件二（测试集）.xlsx')

def extract_test_features(test_data):
    features = []
    for i in range(test_data.shape[0]):
        flux_density = test_data.iloc[i, 5:].values  # Extract flux density data from column 5 onwards
        flux_density = flux_density.astype(float)  # Ensure the data is in float format
        fft_features = extract_fft_features(flux_density)
        features.append(fft_features)
    return np.array(features)

X_test_features_2 = extract_test_features(test_data_2)
y_test_pred_2 = clf.predict(X_test_features_2)

# Save the results to '附件四.xlsx'
test_data_2['波形类型'] = y_test_pred_2
test_data_2['波形类型'] = test_data_2['波形类型'].map({'正弦波': 1, '三角波': 2, '梯形波': 3})
test_data_2.to_excel('附件四.xlsx', index=False)

# Step 6: Predict on 附件三（测试集）
test_data_3 = pd.read_excel('附件三（测试集）.xlsx')
X_test_features_3 = extract_test_features(test_data_3)

# If 附件三 has true labels, use them for accuracy calculation
if '励磁波形' in test_data_3.columns:
    y_true_3 = test_data_3['励磁波形'].values
    y_test_pred_3 = clf.predict(X_test_features_3)
    from sklearn.metrics import accuracy_score
    accuracy_3 = accuracy_score(y_true_3, y_test_pred_3)
    print(f'Accuracy on 附件三: {accuracy_3 * 100:.2f}%')

    # Plot classification accuracy (for 附件三)
    import matplotlib.pyplot as plt
    plt.bar(['附件三'], [accuracy_3 * 100], color='blue')
    plt.ylabel('Accuracy (%)')
    plt.title('Classification Accuracy on 附件三')
    plt.show()

