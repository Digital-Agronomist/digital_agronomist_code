# This script is only to verify that the imputation methods are being replicated 
# correctly based on the selected seed and not that random imputations are being 
# generated each time the imputation is run.

import pandas as pd

# Load the CSV files you want to compare
df1 = pd.read_csv('soil_etl/files/imputation/soil_imputed_KNN.csv')
df2 = pd.read_csv('soil_etl/files/imputation/soil_imputed_KNN1.csv')

# Compare DataFrames to find matches and differences
comparison = df1 == df2

# Calculate the percentage of equality
equal_percentage = comparison.all(axis = 1).mean() * 100

# Calculate the percentage of differences
difference_percentage = 100 - equal_percentage

print(f'Porcentaje de igualdad: {equal_percentage}%')
print(f'Porcentaje de diferencias: {difference_percentage}%')