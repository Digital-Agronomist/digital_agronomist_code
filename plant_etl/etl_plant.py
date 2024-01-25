import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------------------------------------
## 1. Import dataframe

# Import dataframe from Soil and Plant database - Master2.xlsx
corn_v0 = pd.read_excel('plant_etl/Soil and Plant database - Master2.xlsx', sheet_name=3, header=1)

# Remove non-useful columns
corn_v0.drop(corn_v0.columns[2:33], axis=1, inplace=True)

# Dictionary with old and new column names
rename_columns = {
    'B.1': 'B',
    'Mg.1': 'Mg',
    'P.1': 'P',
    'S.1': 'S',
    'K.1': 'K',
    'Ca.2': 'Ca',
    'Mn.1': 'Mn',
    'Fe.1': 'Fe',
    'Cu.1': 'Cu',
    'Zn.1': 'Zn'
}

# Rename columns in the dataframe
corn_v0.rename(columns=rename_columns, inplace=True)

# -------------------------------------------------------------------------------------------------------------
## 2. Extract samples information

# Identify the second column
sample_code = corn_v0.columns[1]

# Filter the rows where the second column starts with 'Corn'
corn_filtered = corn_v0[corn_v0[sample_code].str.startswith('Corn', na=False)]

# Create a new DataFrame with these values
corn_df = pd.DataFrame(corn_filtered[sample_code])

# -------------------------------------------------------------------------------------------------------------
## 3. Create "sample" and "rep" columns

# Remove the first 4 characters from each entry in 'Sample code'
corn_df['Sample code'] = corn_df['Sample code'].str[4:]

# Extract the first characters from each entry in 'Sample code' and create the new columns 'sample' and 'rep'
corn_df['Sample code'] = corn_df['Sample code'].str.strip()
corn_df['sample'] = corn_df['Sample code'].str[:3]
corn_df['rep'] = corn_df['Sample code'].str[6:7]

# Drop the 'Sample code' column as it's the same for all rows and not needed
corn_df.drop('Sample code', axis=1, inplace=True)

# Replace 'rep' with 3 and any empty strings or NaN with 0
corn_df['rep'] = corn_df['rep'].replace('r', '3').replace('', '0').fillna('0')
corn_df['rep'] = corn_df['rep'].replace('R', '3').replace('', '0')

# Convert 'rep' and 'sample' to integers
corn_df['rep'] = corn_df['rep'].astype(int)
corn_df['sample'] = corn_df['sample'].astype(int)

# -------------------------------------------------------------------------------------------------------------
## 4. Extract the nutrient variables from the original dataset and add them to the corn_df dataframe.

# Columns to extract
columns_to_extract = ['B', 'Mg', 'P', 'S', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn']

# Extract the required columns
extracted_columns = corn_filtered[columns_to_extract] # corn_filtered was defined in the line 17

# Adding the extracted columns to corn_df
corn_df = pd.concat([corn_df, extracted_columns], axis=1)

# Save dataframe as corn_df_v1 (first version)
corn_df.to_csv('plant_etl/corn_df_v1.csv', index=False)

# -------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------break-----------------------------------------------------
## 5. N, C, and C/N ratio extraction from Elementar sheet.

# Import dataframe from Soil and Plant database - Master2.xlsx
elementar_v0 = pd.read_excel('plant_etl/Soil and Plant database - Master2.xlsx', sheet_name=6, header=1)

# Dictionary with old and new column names
rename_elementar_columns = {
    'N [%]': 'N',
    'C [%]': 'C',
    'C/N ratio': 'C/N'
}

# Rename columns in the dataframe
elementar_v0.rename(columns=rename_elementar_columns, inplace=True)

# -------------------------------------------------------------------------------------------------------------
## 6. Extract samples information for elementar dataframe

# Identify the second column
sample_code_elementar = elementar_v0.columns[1]

# Filter the rows where the second column starts with 'Corn'
elementar_filtered = elementar_v0[elementar_v0[sample_code_elementar].str.startswith('Corn', na=False)]

# Create a new DataFrame with these values
elementar_df = pd.DataFrame(elementar_filtered[sample_code_elementar])

# -------------------------------------------------------------------------------------------------------------
## 7. Create "sample" and "rep" columns for elementar dataframe

# Remove the first 7 characters from each entry in 'Name'
elementar_df['Name'] = elementar_df['Name'].str[5:]

# Extract the first characters from each entry in 'Name' and create the new columns 'sample' and 'rep'
elementar_df['Name'] = elementar_df['Name'].str.strip()
elementar_df['sample'] = elementar_df['Name'].str[:3]
elementar_df['rep'] = elementar_df['Name'].str[6:7]

# Drop the 'Name' column as it's the same for all rows and not needed
elementar_df.drop('Name', axis=1, inplace=True)

# Replace 'rep' with 3 and any empty strings or NaN with 0
elementar_df['rep'] = elementar_df['rep'].replace('r', '3').replace('', '0').fillna('0')
elementar_df['rep'] = elementar_df['rep'].replace('R', '3').replace('', '0')

# Convert 'rep' and 'sample' to integers
corn_df['rep'] = corn_df['rep'].astype(int)
corn_df['sample'] = corn_df['sample'].astype(int)



print(elementar_df)