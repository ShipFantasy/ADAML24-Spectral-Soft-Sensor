import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files into DataFrames
data1 = pd.read_csv('data_part_1.csv')
data2 = pd.read_csv('data_part_2.csv')

# Check the shape (number of rows and columns) of both datasets
print(data1.shape)
print(data2.shape)

# Check the column names of both datasets
columns_data1 = set(data1.columns)
columns_data2 = set(data2.columns)

# Find differences in column names
only_in_data1 = columns_data1 - columns_data2
only_in_data2 = columns_data2 - columns_data1

print(f"Only in data1: {only_in_data1}")
print(f"Only in data2: {only_in_data2}")

# Get common columns between the two datasets
common_columns = list(set(data1.columns) & set(data2.columns))

# Extract common columns in the order they appear in data1
common_columns = [col for col in data1.columns if col in common_columns]

data1_aligned = data1[common_columns]
data2_aligned = data2[common_columns]

# Merge the two datasets
merged_data = pd.concat([data1_aligned, data2_aligned], ignore_index=True)
merged_data = merged_data.drop(merged_data.columns[0], axis=1)

# Check the shape of the merged dataset
print(merged_data.shape)
print(f"Number of observations: {merged_data.shape[0]}")

# Select the traits to display
data_to_plot = merged_data.iloc[:, 0:20]  

# Display statistical summary of the traits
summary_stats = data_to_plot.describe()
summary_stats = summary_stats.transpose()
print(summary_stats)

# Plot histplots for each trait
fig, axes = plt.subplots(4, 5, figsize=(20, 15))  
axes = axes.flatten()  

for i, col in enumerate(data_to_plot.columns):
    axes[i].hist(data_to_plot[col], bins=30, color='blue', alpha=0.7)  
    axes[i].set_title(f'{col}')  
plt.tight_layout()
plt.show()

# Plot boxplots for each trait
fig, axes = plt.subplots(4, 5, figsize=(20, 10))  
axes = axes.flatten()

for i, col in enumerate(data_to_plot.columns):
    data_to_plot[col].plot(kind='box', ax=axes[i], title=f'{col}') 

plt.tight_layout()
plt.show()

# Select spectral data for plotting
subset_data = merged_data.iloc[:, 20:]  # Band data

# Plot heatmap of spectral data
plt.figure(figsize=(12, 8))
sns.heatmap(subset_data, cmap='viridis', cbar=True)
plt.title('Heatmap of Hyperspectral Data')
plt.xlabel('Bands')
plt.ylabel('Samples')
plt.show()

# Plotting superimposed spectral curves for the first 10 samples
samples_to_plot = merged_data.iloc[:10, 20:]  
wavelengths = merged_data.columns[20:]

plt.figure(figsize=(10, 6))

for i, row in samples_to_plot.iterrows():
    plt.plot(wavelengths, row, label=f'Sample {i+1}')

plt.title('Overlay of Spectral Curves for First 10 Samples')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend(loc='upper right')
plt.show()
