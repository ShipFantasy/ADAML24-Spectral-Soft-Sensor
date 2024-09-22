import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

# Import and merge the two datasets
data1 = pd.read_csv('data_part_1.csv')
data2 = pd.read_csv('data_part_2.csv')

# Get common columns between the two datasets
common_columns = list(set(data1.columns) & set(data2.columns))
common_columns = [col for col in data1.columns if col in common_columns]

data1_aligned = data1[common_columns]
data2_aligned = data2[common_columns]

merged_data = pd.concat([data1_aligned, data2_aligned], ignore_index=True)
data = merged_data.drop(merged_data.columns[0], axis=1)

# Check the shape of the merged dataset
print(data.shape)

y = data.iloc[:, :20] 
X = data.iloc[:, 20:]  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()

# Fit the scaler on the training set and transform the data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the validation and test sets
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

missing_values = X.isnull().sum()

# Filter out the features that have missing values
missing_features = missing_values[missing_values > 0] 

print(missing_features)

# Z-scores
z_scores = np.abs(stats.zscore(X_train_scaled))
outliers = np.where(z_scores > 3)

# Get unique indices of outlier rows (samples)
outlier_rows = np.unique(outliers[0])

# Randomly select 10 outlier samples
selected_outlier_samples = np.random.choice(outlier_rows, size=10, replace=False)

# Plot these samples and highlight their outliers
plt.figure(figsize=(15, 6))

for i, sample_idx in enumerate(selected_outlier_samples):
    plt.subplot(2, 5, i+1)
    
    # Plot all feature values for the sample
    plt.scatter(range(X_train_scaled.shape[1]), X_train_scaled[sample_idx, :], alpha=0.6)
    
    # Highlight outlier values in red
    sample_outlier_cols = outliers[1][outliers[0] == sample_idx]
    plt.scatter(sample_outlier_cols, X_train_scaled[sample_idx, sample_outlier_cols], color='red')

    plt.title(f"Sample {sample_idx}")
    plt.xlabel('Feature Index')
    plt.ylabel('Scaled Value')

plt.tight_layout()
plt.show()

# Apply Savitzky-Golay filter to smooth the spectrum data for each sample
# Set window size and polynomial order
window_size = 45  
poly_order = 3

X_train_smoothed = savgol_filter(X_train_scaled, window_length=window_size, polyorder=poly_order, axis=1)


# Compare before and after smoothing curve

# Randomly select 6 sample indices
sample_indices = np.random.choice(X_train_scaled.shape[0], 6, replace=False)

# Plot comparison of original and smoothed data
plt.figure(figsize=(12, 8))

for i, sample_idx in enumerate(sample_indices, 1):
    plt.subplot(3, 2, i)
    plt.plot(range(X_train_scaled.shape[1]), X_train_scaled[sample_idx, :], label='Original', alpha=0.6)
    plt.plot(range(X_train_scaled.shape[1]), X_train_smoothed[sample_idx, :], label='Smoothed', alpha=0.8, linestyle='--')
    plt.title(f'Sample {sample_idx}: Original vs Smoothed Spectrum')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Add extra space between subplots
plt.show()

valid_data_counts = y_train.notna().sum()
top_5_features = valid_data_counts.nlargest(5).index

print("Top 5 target features based on valid data counts:", top_5_features)

y_train_top_5 = y_train[top_5_features]

# Create a dictionary to store X and y for each target feature, only including samples without missing values
data_storage = {}

# Iterate through each target feature
for target in y_train_top_5.columns:
    valid_indices = y_train_top_5[target].notna()
    
    # Extract the corresponding X and y, excluding samples with missing values
    X_valid = X_train_smoothed[valid_indices]
    y_valid = y_train_top_5[target][valid_indices]
    
    data_storage[target] = {
        'X': X_valid,  
        'y': y_valid   
    }
    
    X_shape = X_valid.shape
    y_shape = y_valid.shape
    print(f"Target: {target} -> X shape: {X_shape}, y shape: {y_shape}")

# Output the data storage structure
print(f"Data storage keys: {list(data_storage.keys())}")

for target, data in data_storage.items():
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)  # Data after dimensionality reduction
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Visualize the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6)
    plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Explained Variance by PCs for {target}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(f"Target: {target} -> Explained variance (first few components): {explained_variance_ratio[:5]}")
    print(f"Target: {target} -> Cumulative explained variance (first few components): {cumulative_variance[:5]}")

    