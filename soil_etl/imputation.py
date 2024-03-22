import pandas as pd
import numpy as np
# Exploring missing data
import missingno as msno
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from scipy.stats import shapiro
# For Expectation-Maximization
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
# For K-Nearest Neighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
# Validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ---------------------------------------------------------------------------------------
## 0. Import and transform dataframe 
soil_mix_df_final = pd.read_csv('soil_mix_df_final.csv')

# Delete columns that are not necessary
soil_mix_cleaned = soil_mix_df_final.drop(['soil','sample', 'rep'], axis = 1)
soil_mix_cleaned2 = soil_mix_df_final.drop(['sample', 'rep'], axis = 1)

# Replace zeros with NaN for imputation
soil_mix_cleaned.replace(0, np.nan, inplace = True)
soil_mix_cleaned2.replace(0, np.nan, inplace = True)

# Extract subset without missing values
complete_cases = soil_mix_cleaned.dropna()
complete_cases2 = soil_mix_cleaned2.dropna()

# Perform MANOVA to confirm if there is a relationship between the variable 'soil' 
# and other variables
maov = MANOVA.from_formula('B + Mg + P + S + K + Ca + Mn + Fe + Cu + Zn ~ soil', data = complete_cases2)
print(maov.mv_test())

# According to the result, there is a significant relationship between the "soil" variable 
# and the nutrient variables in the dataframe of complete cases 

# ---------------------------------------------------------------------------------------
# -----------------------------------------break-----------------------------------------
# ----------------------------------Exploring missin data--------------------------------
## 1. Graphic exploring missing data

# Missing data matrix
msno.matrix(soil_mix_cleaned2)
plt.title('Missing data matrix')
plt.show()

# ---------------------------------------------------------------------------------------
## 2. Analysis of the Nature of Missing Data

# Simplify the problem by dealing with binary data for missing or non-missing
df_missing = soil_mix_cleaned2.isnull().astype(int)

# List to save the results of logistic regressions
regression_results = []

# Iterate over each column to treat it as the dependent variable in a logistic regression
for col in df_missing.columns:
    # Prepare the data set for regression, excluding the current column
    X = df_missing.drop(col, axis=1)
    y = df_missing[col]
    
    # Ensure that we are not using an empty column (no missing values)
    if y.sum() == 0:
        continue
    
    # Impute missing values in X for regression with the mean (this is just so we can fit the model)
    imp = SimpleImputer(missing_values=1, strategy='mean')
    X_imputed = imp.fit_transform(X)
    
    # Fit logistic regression
    model = LogisticRegression(solver='liblinear')
    model.fit(X_imputed, y)
    
    # Calculate log loss
    y_pred = model.predict_proba(X_imputed)
    loss = log_loss(y, y_pred)
    
    # Save the results
    regression_results.append((col, loss))

# Print the results
for result in regression_results:
    print(f'Variable: {result[0]}, Log Loss: {result[1]}')

# Variables like P and Cu, with very low log losses 
# (0.013690905931761721 and 0.0519370894042759, respectively), could suggest that 
# the patterns of missing data in these variables are more clearly related to the other 
# variables in your data set.

# Variables such as B and S, with relatively high log losses 
# (0.6622881367686035 and 0.6779959105580132), could suggest that the patterns of 
# missing data in these variables are less predictable based on the other variables, 
# which could indicate MCAR data or simply reflect a poor fit of the model.

# ---------------------------------------------------------------------------------------
## 3. Percentage of missing values
missing_percentage = soil_mix_cleaned2.isnull().mean() * 100
print(missing_percentage)

# ---------------------------------------------------------------------------------------
## 4. Distribution analysis   
for column in soil_mix_cleaned.columns:
    if soil_mix_cleaned[column].isna().sum() < len(soil_mix_cleaned) * 0.65:  # With this threshold all variables can be included
        # Histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(soil_mix_cleaned[column].dropna(), kde=True)
        plt.title(f'Histograma de {column}')
        plt.show()

        # Shapiro-Wilk test
        stat, p = shapiro(soil_mix_cleaned[column].dropna())
        print(f'Prueba de Shapiro-Wilk para {column}: Estadístico={stat}, p-valor={p}')

# None of the variables evaluated follow a normal distribution based on the p-values 
# of the Shapiro-Wilk test. This is important because it affects the choice of statistical 
# methods for subsequent analysis. Many statistical tests and models assume normality in 
# the data. Non-normality may require that nonparametric techniques be used or that the 
# data be transformed to approximate a normal distribution before analysis.

# These results also inform how to address imputation of missing data. For example, 
# imputation by the mean would not be appropriate for data that is not normal, since 
# the mean is sensitive to skewed data and outliers.

# ---------------------------------------------------------------------------------------
# -----------------------------------------break-----------------------------------------
# -------------------------Imputation only with numerical values-------------------------
## 1. Imputation using Expectation-Maximization (EM) 

# EM using sklearn's IterativeImputer, considering NAs now as NaN
imputer = IterativeImputer(max_iter = 10, random_state = 1)
soil_imputed_EM = imputer.fit_transform(soil_mix_cleaned)
soil_imputed_EM = pd.DataFrame(soil_imputed_EM, columns = soil_mix_cleaned.columns)
print(soil_imputed_EM)

# ---------------------------------------------------------------------------------------
## 2. Imputation using K-Nearest Neighbors (KNN) 

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

# Set the seed
seed_value = 5
torch.manual_seed(seed_value)
np.random.seed(seed_value)

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

# Set the seed
np.random.seed(6)

# Imputation
soil_imputed_svd = SoftImpute().fit_transform(soil_mix_cleaned)
soil_imputed_svd = pd.DataFrame(soil_imputed_svd, columns = soil_mix_cleaned.columns)
print(soil_imputed_svd)

# ---------------------------------------------------------------------------------------
## 9. Save imputed datasets
# soil_imputed_EM.to_csv('files/imputation/soil_imputed_EM.csv', index = False)
# soil_imputed_KNN.to_csv('files/imputation/soil_imputed_KNN.csv', index = False)
# soil_imputed_linear.to_csv('files/imputation/soil_imputed_linear.csv', index = False)
# soil_imputed_mice.to_csv('files/imputation/soil_imputed_mice.csv', index = False)
# soil_imputed_NN.to_csv('files/imputation/soil_imputed_NN.csv', index = False)
# soil_imputed_rf.to_csv('files/imputation/soil_imputed_rf.csv', index = False)
# soil_imputed_svd.to_csv('files/imputation/soil_imputed_svd.csv', index = False)

# ---------------------------------------------------------------------------------------
# -----------------------------------------break-----------------------------------------
# --------------------------Imputation only with 'soil' column---------------------------

## 0. Preprocessing
numerical_variables = soil_mix_cleaned2.select_dtypes(include = [np.number])
categorical_variables = soil_mix_cleaned2.select_dtypes(exclude = [np.number])

# Apply One-Hot Encoding to Categorical Column
encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
categorical_variables_encoded = encoder.fit_transform(categorical_variables)

# Join the coded numerical and categorical variables again
data_for_imputation = np.hstack([numerical_variables, categorical_variables_encoded])

# ---------------------------------------------------------------------------------------
## 1. Imputation using Expectation-Maximization (EM)

# Apply EM imputation
imputer = IterativeImputer(max_iter = 10, random_state = 1)
data_imputed = imputer.fit_transform(data_for_imputation)

# Convert back to DataFrame
columns = list(numerical_variables.columns) + list(encoder.get_feature_names_out(categorical_variables.columns))
soil_imputed_EM2 = pd.DataFrame(data_imputed, columns = columns)

# List of One-Hot columns coded for the variable 'soil'
one_hot_columns = [col for col in soil_imputed_EM2.columns if col.startswith('soil_')]

# Convert One-Hot columns back to a categorical column
# Identify the dummy column with the maximum value in each row (predicted category)
soil_imputed_EM2['soil_predicted'] = soil_imputed_EM2[one_hot_columns].idxmax(axis = 1)

# Extract original category name from One-Hot column names
soil_imputed_EM2['soil_predicted'] = soil_imputed_EM2['soil_predicted'].apply(lambda x: x.split('_')[1])

# 'soil_predicted' now contains the 'soil' categories reconstructed after imputation
soil_imputed_EM2 = soil_imputed_EM2.drop(columns = one_hot_columns)

# Rename 'soil_predicted' column to 'soil'
soil_imputed_EM2.rename(columns={'soil_predicted': 'soil'}, inplace = True)

# Next, we want to move the 'soil' column to the beginning of the DataFrame
# We create a list of the columns, putting 'soil' at the beginning
columns = ['soil'] + [col for col in soil_imputed_EM2.columns if col != 'soil']

# Reorder the DataFrame to reflect this new column order
soil_imputed_EM2 = soil_imputed_EM2[columns]
print(soil_imputed_EM2)

# ---------------------------------------------------------------------------------------
## 2. Imputation using K-Nearest Neighbors (KNN) 

# Standardization and KNN Imputation as a pipeline
pipeline = Pipeline(steps = [('scale', StandardScaler()), 
                            ('impute', KNNImputer(n_neighbors = 5, weights = "uniform"))])

# Apply the pipeline to the combined set
data_imputed_scaled = pipeline.fit_transform(data_for_imputation)

# Reverse standardization to obtain the imputed data set on its original scale
scaler = StandardScaler().fit(numerical_variables)  # Adjust the scaler to numeric variables only
data_imputed = scaler.inverse_transform(data_imputed_scaled[:, :len(numerical_variables.columns)])  # Reverse standardization only in the numerical parts

# Retrieve categorical data from the imputed and scaled set (without needing to revert standardization on categorical data)
data_categorical_imputed = data_imputed_scaled[:, len(numerical_variables.columns):]

# Convert One-Hot Encoding Back to Categorical Labels
categorical_reversed = encoder.inverse_transform(data_categorical_imputed)

# Create DataFrames from numpy arrays to facilitate concatenation
df_numericas_imputed = pd.DataFrame(data_imputed, columns = numerical_variables.columns)
df_categorical_reversed = pd.DataFrame(categorical_reversed, columns = categorical_variables.columns)

# Concatenate the imputed numerical variables and the reversed categorical variables
soil_imputed_KNN2 = pd.concat([df_categorical_reversed.reset_index(drop = True), df_numericas_imputed.reset_index(drop = True)], axis=1)
print(soil_imputed_KNN2)


# ---------------------------------------------------------------------------------------
# -----------------------------------------break-----------------------------------------
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

##### Apparently the best is between SVD and NN, although it is difficult to say

# ---------------------------------------------------------------------------------------
## 2. Correlation analysis

# Calculate the original correlation matrix
correlation_original = soil_mix_cleaned.corr()

# Generate correlation diagrams for each imputation method
for label, df in dataframes.items():
    correlation_imputed = df.corr()
    plt.figure(figsize = (10, 8))
    sns.heatmap(correlation_original - correlation_imputed, cmap = 'coolwarm', center=0)
    plt.title(f'Differences in Correlation for {label} Imputation')
    plt.show()

# Conclusion: When evaluating the graphs, a good imputation method would have most cells 
# close to zero (neutral color on the graph), indicating that there is not much difference 
# between the original correlations and the correlations after imputation.

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
best_method = min(sum_abs_differences, key = sum_abs_differences.get)

print("Suma de diferencias absolutas por método:", sum_abs_differences)
print("El mejor método de imputación es:", best_method, "con una suma de diferencias de:", sum_abs_differences[best_method])

# ---------------------------------------------------------------------------------------
## 3. Validation by comparing with a complete subset

# Iterar sobre cada columna del DataFrame completo para generar los gráficos
for column in complete_cases.columns:
    plt.figure(figsize=(10, 6))
    # Graficar la distribución del subconjunto completo para la columna actual
    sns.kdeplot(complete_cases[column], label='Complete Cases', linestyle='--')
    # Graficar la distribución de la columna para cada método de imputación
    for label, df in dataframes.items():
        sns.kdeplot(df[column], label=f'Imputed with {label}')
    plt.title(f'Distribution Comparison for {column}')
    plt.legend()
    plt.show()
