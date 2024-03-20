import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# For Expectation-Maximization
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# For K-Nearest Neighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
# For Linea Regression
from sklearn.linear_model import LinearRegression
# For Random Forest
from sklearn.ensemble import RandomForestRegressor
# For MICE
from sklearn.linear_model import BayesianRidge
# For Neural Network (PyTorch)
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
# For SVD
from fancyimpute import SoftImpute
# For Matrix Factorization
from fancyimpute import MatrixFactorization

# ---------------------------------------------------------------------------------------
## 1. Import and transform dataframe 

soil_mix_df_final = pd.read_csv('soil_mix_df_final.csv')

# Delete the first 3 columns that are not necessary
soil_mix_cleaned = soil_mix_df_final.drop(['soil', 'sample', 'rep'], axis = 1)

# Replace zeros with NaN for imputation
soil_mix_cleaned.replace(0, np.nan, inplace = True)
print(soil_mix_cleaned)

# ---------------------------------------------------------------------------------------
## 2. Imputation using Expectation-Maximization (EM) 

# EM using sklearn's IterativeImputer, considering NAs now as NaN
imputer = IterativeImputer(max_iter = 10, random_state = 1)
soil_imputed_EM = imputer.fit_transform(soil_mix_cleaned)
soil_imputed_EM = pd.DataFrame(soil_imputed_EM, columns = soil_mix_cleaned.columns)
print(soil_imputed_EM)

# ---------------------------------------------------------------------------------------
## 3. Imputation using K-Nearest Neighbors (KNN) 

# It is advisable to standardize data for KNN
scaler = StandardScaler()
soil_mix_scaled = scaler.fit_transform(soil_mix_cleaned)

# Imputation
imputer_knn = KNNImputer(n_neighbors = 5, weights = "uniform")
soil_imputed_KNN = imputer_knn.fit_transform(soil_mix_scaled)

# Convert back to DataFrame and undo standardization
soil_imputed_KNN = pd.DataFrame(scaler.inverse_transform(soil_imputed_KNN), columns = soil_mix_cleaned.columns)
print(soil_imputed_KNN)

# ---------------------------------------------------------------------------------------
## 4. Imputation using Linear Regression

imputer_linear = IterativeImputer(estimator=LinearRegression(), max_iter = 10, random_state = 2)
soil_imputed_linear = imputer_linear.fit_transform(soil_mix_cleaned)

# Convert back to DataFrame
soil_imputed_linear = pd.DataFrame(soil_imputed_linear, columns = soil_mix_cleaned.columns)
print(soil_imputed_linear)

# ---------------------------------------------------------------------------------------
## 5. Imputation using Random Forest Regression

imputer_rf = IterativeImputer(estimator = RandomForestRegressor(n_estimators = 100, random_state = 3), max_iter=10, random_state=0)
soil_imputed_rf = imputer_rf.fit_transform(soil_mix_cleaned)

# Convert back to DataFrame
soil_imputed_rf = pd.DataFrame(soil_imputed_rf, columns = soil_mix_cleaned.columns)
print(soil_imputed_rf)

# ---------------------------------------------------------------------------------------
## 6. Imputation using Multiple Imputation by Chained Equations (MICE)

# MICE imputation with a BayesianRidge model as an estimator
imputer_mice = IterativeImputer(estimator = BayesianRidge(), max_iter = 10, random_state = 4)
soil_imputed_mice = imputer_mice.fit_transform(soil_mix_cleaned)
soil_imputed_mice = pd.DataFrame(soil_imputed_mice, columns = soil_mix_cleaned.columns)
print(soil_imputed_mice)

# ---------------------------------------------------------------------------------------
## 7. Imputation using Neural Network

# Step 1: Data Preparation
# Data normalization (example with MinMaxScaler to keep data between 0 and 1)
scaler2 = MinMaxScaler()
soil_mix_scaled2 = soil_mix_cleaned.copy()
soil_mix_scaled2[soil_mix_scaled2.columns] = scaler2.fit_transform(soil_mix_scaled2[soil_mix_scaled2.columns])

# Convert NaN to 0 for training
soil_mix_scaled2_numpy = soil_mix_scaled2.fillna(0).to_numpy()
soil_mask_numpy = ~soil_mix_cleaned.isna().to_numpy()  # Mask where NaN values were from

# Convert to tensors
data_tensor = torch.tensor(soil_mix_scaled2_numpy, dtype = torch.float32)
mask_tensor = torch.tensor(soil_mask_numpy, dtype = torch.float32)

# Step 2: Definition of the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Sigmoid()  # Use sigmoid if data is normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder(num_features = soil_mix_scaled2.shape[1])

# Step 3: Training the Autoencoder
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs * mask_tensor, data_tensor * mask_tensor)  # Calculate loss only on non-NaN data
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Step 4: Imputation of Missing Data

# Make sure 'soil_imputed_NN' is set to a copy of 'soil_mix_cleaned'
soil_imputed_NN = soil_mix_cleaned.copy()

# Reversal of normalization to obtain the imputed values on their original scale
with torch.no_grad():
    imputed_data = model(data_tensor).numpy()

imputed_data_original_scale = scaler2.inverse_transform(imputed_data)

# Create a DataFrame with the imputed data to facilitate the assignment
imputed_df = pd.DataFrame(imputed_data_original_scale, columns = soil_mix_cleaned.columns)

# Replace the NaN values in the original DataFrame with the imputed values
for column in soil_mix_cleaned.columns:
    soil_imputed_NN[column].loc[soil_mix_cleaned[column].isna()] = imputed_df[column].loc[soil_mix_cleaned[column].isna()]

# Print soil_imputed_NN
print(soil_imputed_NN)

# ---------------------------------------------------------------------------------------
## 8. Imputation using Singular Value Decomposition (SVD)
soil_imputed_svd = SoftImpute().fit_transform(soil_mix_cleaned)
soil_imputed_svd = pd.DataFrame(soil_imputed_svd, columns = soil_mix_cleaned.columns)
print(soil_imputed_svd)

# ---------------------------------------------------------------------------------------
# -----------------------------------------break----------------------------------------------
# ----------------------------------Imputation validation--------------------------------

## 1. Comparison of distributions

dataframes = {
    'EM': soil_imputed_EM,
    'KNN': soil_imputed_KNN,
    'LR': soil_imputed_linear,
    'MICE': soil_imputed_mice,
    'NN': soil_imputed_NN,
    'RF': soil_imputed_rf,
    'SVD': soil_imputed_svd,
}

# Prepare a DataFrame for the combined Z-scores
z_combined = pd.DataFrame()
z_combined['Original'] = soil_mix_cleaned.apply(zscore, axis = 0).values.flatten()

# Apply zscore and flatten values for imputed DataFrames
for label, df in dataframes.items():
    z_combined[label] = df.apply(zscore, axis = 0).values.flatten()

# Plot the combined Z-score distribution for the original with specific styles
plt.figure(figsize=(15, 10))
sns.kdeplot(z_combined['Original'], label = 'Original', color = 'black', linestyle = '--')

# Now graph the other imputed DataFrames
for label in dataframes.keys():
    sns.kdeplot(z_combined[label], label = label)

plt.legend()
plt.title('Combined Z-Score Distributions of All Variables')
plt.show()

##### Apparently the best is MICE, although it is difficult to say

# ---------------------------------------------------------------------------------------
## 2. Correlation analysis

# Calculate the original correlation matrix
correlation_original = soil_mix_cleaned.corr()

# Generate correlation diagrams for each imputation method
for label, df in dataframes.items():
    correlation_imputed = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_original - correlation_imputed, cmap='coolwarm', center=0)
    plt.title(f'Differences in Correlation for {label} Imputation')
    plt.show()

# Conclusion: When evaluating the graphs, a good imputation method would have most cells 
# close to zero (neutral color on the graph), indicating that there is not much difference 
# between the original correlations and the correlations after imputation.
# Graphically, the best method is between: KNN, NN y MICE.

# Calculate and save correlation matrices
for label, df in dataframes.items():
    correlation = df.corr()
    filepath = f'soil_etl/files/imputation/correlation_{label}.csv'
    correlation.to_csv(filepath)

# Dictionary to store the sums of absolute differences
sum_abs_differences = {}

for label, df in dataframes.items():
    # Calculate correlation matrix for imputed DataFrame
    correlation_imputed = df.corr()
    # Calculate the absolute difference between the original and the imputed matrix
    abs_difference = np.abs(correlation_original - correlation_imputed)
    # Add all absolute differences to obtain a discrepancy measure
    sum_abs_differences[label] = np.sum(abs_difference.values)

# Find the method with the smallest sum of absolute differences
best_method = min(sum_abs_differences, key=sum_abs_differences.get)

print("Suma de diferencias absolutas por método:", sum_abs_differences)
print("El mejor método de imputación es:", best_method, "con una suma de diferencias de:", sum_abs_differences[best_method])

