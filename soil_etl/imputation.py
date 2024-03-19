import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

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
# For Neural Network


# ---------------------------------------------------------------------------------------
## 1. Import and transform dataframe 

# Suponiendo que df es tu DataFrame
soil_mix_df_final = pd.read_csv('soil_mix_df_final.csv')

# Eliminar las primeras 3 columnas que no son necesarias
soil_mix_cleaned = soil_mix_df_final.drop(['soil', 'sample', 'rep'], axis = 1)

# Reemplazar ceros por NaN para la imputación
soil_mix_cleaned.replace(0, np.nan, inplace = True)
print(soil_mix_cleaned)

# ---------------------------------------------------------------------------------------
## 2. Imputation using Expectation-Maximization (EM) 

# EM usando IterativeImputer de sklearn, considerando los NAs ahora como NaN
imputer = IterativeImputer(max_iter = 10, random_state = 1)
soil_imputed_EM = imputer.fit_transform(soil_mix_cleaned)
soil_imputed_EM = pd.DataFrame(soil_imputed_EM, columns = soil_mix_cleaned.columns)
print(soil_imputed_EM)

# ---------------------------------------------------------------------------------------
## 3. Imputation using K-Nearest Neighbors (KNN) 

# Es recomendable estandarizar los datos para KNN
scaler = StandardScaler()
soil_mix_scaled = scaler.fit_transform(soil_mix_cleaned)

#Imputation
imputer_knn = KNNImputer(n_neighbors = 5, weights = "uniform")
soil_imputed_KNN = imputer_knn.fit_transform(soil_mix_scaled)

# Convertir de nuevo a DataFrame y deshacer la estandarización
soil_imputed_KNN = pd.DataFrame(scaler.inverse_transform(soil_imputed_KNN), columns = soil_mix_cleaned.columns)
print(soil_imputed_KNN)

# ---------------------------------------------------------------------------------------
## 4. Imputation using Linear Regression

# Asumiendo que soil_mix_cleaned es tu DataFrame después de eliminar las primeras 3 columnas y reemplazar 0s por NaNs

imputer_linear = IterativeImputer(estimator=LinearRegression(), max_iter = 10, random_state = 2)
soil_imputed_linear = imputer_linear.fit_transform(soil_mix_cleaned)

# Convertir de nuevo a DataFrame
soil_imputed_linear = pd.DataFrame(soil_imputed_linear, columns = soil_mix_cleaned.columns)
print(soil_imputed_linear)

# ---------------------------------------------------------------------------------------
## 5. Imputation using Random Forest Regression

imputer_rf = IterativeImputer(estimator = RandomForestRegressor(n_estimators = 100, random_state = 3), max_iter=10, random_state=0)
soil_imputed_rf = imputer_rf.fit_transform(soil_mix_cleaned)

# Convertir de nuevo a DataFrame
soil_imputed_rf = pd.DataFrame(soil_imputed_rf, columns = soil_mix_cleaned.columns)
print(soil_imputed_rf)

# ---------------------------------------------------------------------------------------
## 6. Imputation using Multiple Imputation by Chained Equations (MICE)

# Imputación MICE con un modelo BayesianRidge como estimador
imputer_mice = IterativeImputer(estimator = BayesianRidge(), max_iter = 10, random_state = 4)
soil_imputed_mice = imputer_mice.fit_transform(soil_mix_cleaned)
soil_imputed_mice = pd.DataFrame(soil_imputed_mice, columns=soil_mix_cleaned.columns)
print(soil_imputed_mice)

# ---------------------------------------------------------------------------------------
## 7. Imputation using Neural Network


