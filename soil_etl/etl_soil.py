import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine

from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -------------------------------------------------------------------------------------------------------------
## 0. Connect to the database and extract the data

# Establish the connection
# sqlalchemy uses a standard URL for connections: 
# 'mysql+pymysql://<user>:<password>@<host>/<dbname>'
DATABASE_CON = os.getenv('DATABASE_CON')

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_CON)

# Query to extract the soil_ICP dataframe
soil_icp_df = pd.read_sql_query("""
                        -- This query extracts the information necessary to shape the soil_icp dataframe
                        
                        SELECT sr.id, s.name, sr.sample, sr.rep,
                            MAX(CASE WHEN n.symbol = 'B' THEN rn.value ELSE 0 END) AS B,
                            MAX(CASE WHEN n.symbol = 'Mg' THEN rn.value ELSE 0 END) AS Mg,
                            MAX(CASE WHEN n.symbol = 'P' THEN rn.value ELSE 0 END) AS P,
                            MAX(CASE WHEN n.symbol = 'S' THEN rn.value ELSE 0 END) AS S,
                            MAX(CASE WHEN n.symbol = 'K' THEN rn.value ELSE 0 END) AS K,
                            MAX(CASE WHEN n.symbol = 'Ca' THEN rn.value ELSE 0 END) AS Ca,
                            MAX(CASE WHEN n.symbol = 'Mn' THEN rn.value ELSE 0 END) AS Mn,
                            MAX(CASE WHEN n.symbol = 'Fe' THEN rn.value ELSE 0 END) AS Fe,
                            MAX(CASE WHEN n.symbol = 'Cu' THEN rn.value ELSE 0 END) AS Cu,
                            MAX(CASE WHEN n.symbol = 'Zn' THEN rn.value ELSE 0 END) AS Zn
                        FROM soil_results AS sr
                        JOIN soils AS s ON sr.soil_id = s.id
                        JOIN result_nutrients AS rn ON sr.id = rn.soil_result_id
                        JOIN nutrients AS n ON rn.nutrient_id = n.id
                        WHERE sr.analysis_method_id  = 2
                        GROUP BY sr.id
                        ORDER BY sr.id;""", engine)

# Query to extract the soil_HHXRF dataframe
soil_hhxrf_df = pd.read_sql_query("""
                        -- This query extracts the information necessary to shape the soil_icp dataframe
                        
                        SELECT sr.id, s.name, sr.sample, sr.rep,
                            MAX(CASE WHEN n.symbol = 'B' THEN rn.value ELSE 0 END) AS B,
                            MAX(CASE WHEN n.symbol = 'Mg' THEN rn.value ELSE 0 END) AS Mg,
                            MAX(CASE WHEN n.symbol = 'P' THEN rn.value ELSE 0 END) AS P,
                            MAX(CASE WHEN n.symbol = 'S' THEN rn.value ELSE 0 END) AS S,
                            MAX(CASE WHEN n.symbol = 'K' THEN rn.value ELSE 0 END) AS K,
                            MAX(CASE WHEN n.symbol = 'Ca' THEN rn.value ELSE 0 END) AS Ca,
                            MAX(CASE WHEN n.symbol = 'Mn' THEN rn.value ELSE 0 END) AS Mn,
                            MAX(CASE WHEN n.symbol = 'Fe' THEN rn.value ELSE 0 END) AS Fe,
                            MAX(CASE WHEN n.symbol = 'Cu' THEN rn.value ELSE 0 END) AS Cu,
                            MAX(CASE WHEN n.symbol = 'Zn' THEN rn.value ELSE 0 END) AS Zn
                        FROM soil_results AS sr
                        JOIN soils AS s ON sr.soil_id = s.id
                        JOIN result_nutrients AS rn ON sr.id = rn.soil_result_id
                        JOIN nutrients AS n ON rn.nutrient_id = n.id
                        WHERE sr.analysis_method_id  = 3
                        GROUP BY sr.id
                        ORDER BY sr.id;""", engine)

# Correct index "id" of soil_hhxrf_df so that it starts at 1
soil_hhxrf_df = soil_hhxrf_df.drop('id', axis=1)
soil_hhxrf_df.reset_index(drop=True, inplace=True)
soil_hhxrf_df.index += 1
soil_hhxrf_df['id'] = soil_hhxrf_df.index

# -------------------------------------------------------------------------------------------------------------
## 1. Import dataframes from files
# Import dataframe of soil_ICP
# soil_icp_df = pd.read_csv('soil_etl/soil_ICP.csv')

# Import dataframe of soil_HHXRF
#soil_hhxrf_df = pd.read_csv('soil_etl/soil_HHXRF.csv')

# -------------------------------------------------------------------------------------------------------------
## 2. Creation of a table of adequate ranges of nutrients
# Nutrient range for sandy soils
nutrients_sandy = ['limit','B', 'Mg', 'P', 'S', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn']
sandy_soil_nutrient_range = pd.DataFrame(columns=nutrients_sandy)
sandy_soil_nutrient_range.loc[0] = ['inferior', '0.5', '51', '23', '30', '66', '400', '10', '2.6', '1.0', '3.1']  # Values for the first row
sandy_soil_nutrient_range.loc[1] = ['superior','1.0', '250', '32', '40', '90', '600', '20', '4.7', '5.5', '20']  # Values for the second row
print(sandy_soil_nutrient_range)

# Nutrient range for Medium-textured soils
nutrients_medium = ['limit','B', 'Mg', 'P', 'S', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn']
soil_medium_nutrient_range = pd.DataFrame(columns=nutrients_medium)
soil_medium_nutrient_range.loc[0] = ['inferior', '0.9', '101', '11', '30', '81', '601', '10', '2.6', '1.0', '3.1']  # Values for the first row
soil_medium_nutrient_range.loc[1] = ['superior','1.5', '500', '20', '40', '110', '1000', '20', '4.7', '5.5', '20']  # Values for the second row
print(soil_medium_nutrient_range)

#-------------------------------------------------------------------------------------------------------------------
## 3. Creation of a table of soil types
soil_types = ['soil', 'type']
soil_types_df = pd.DataFrame(columns=soil_types)

# Function to determine the soil type
def determine_soil_type(soil_name):
    if soil_name == 'patrick':
        return 'medium'
    elif soil_name == 'werner':
        return 'medium'
    else:
        return 'sandy'

soil_types_df.loc[0] = ['krm', determine_soil_type('krm')]
soil_types_df.loc[1] = ['lobby', determine_soil_type('lobby')]
soil_types_df.loc[2] = ['yenter', determine_soil_type('yenter')]
soil_types_df.loc[3] = ['pow', determine_soil_type('pow')]
soil_types_df.loc[4] = ['c15', determine_soil_type('c15')]
soil_types_df.loc[5] = ['c21', determine_soil_type('c21')]
soil_types_df.loc[6] = ['c28', determine_soil_type('c28')]
soil_types_df.loc[7] = ['coloma', determine_soil_type('coloma')]
soil_types_df.loc[8] = ['patrick', determine_soil_type('patrick')]
soil_types_df.loc[9] = ['werner', determine_soil_type('werner')]
soil_types_df.loc[10] = ['wormet', determine_soil_type('wormet')]
print(soil_types_df)

# --------------------------------------------------------------------------------------------------------------------
## 4. Checking if the ICP values are in the range

# Initialize the new DataFrame
icp_classification = ['id'] + list(soil_icp_df.columns[1:4]) + nutrients_sandy[1:]
icp_classification_df = pd.DataFrame(columns=icp_classification)

# Iterate over soil_icp_df to fill icp_classification_df
for id, row in soil_icp_df.iterrows():
    new_id = id + 1
    new_row = [new_id] + list(row[1:4])

    # Determine soil type
    soil_name = row[1]
    soil_type = soil_types_df[soil_types_df['soil'] == soil_name]['type'].iloc[0]

    # Select the appropriate nutrient range table
    nutrient_range_df = sandy_soil_nutrient_range if soil_type == 'sandy' else soil_medium_nutrient_range

    # Compare and categorize values
    for element in nutrients_sandy[1:]:
        element_value = row[element]

        if element_value == 0:  # Check if the value is zero
            category = 'NA'  # Assign None (which will be NULL in CSV)
        else:
            lower_limit = float(nutrient_range_df[nutrient_range_df['limit'] == 'inferior'][element])
            upper_limit = float(nutrient_range_df[nutrient_range_df['limit'] == 'superior'][element])

            if element_value < lower_limit:
                category = 'low'
            elif element_value > upper_limit:
                category = 'high'
            else:
                category = 'optimum'
        
        new_row.append(category)

    icp_classification_df.loc[new_id] = new_row

# Export icp_classification_df to a CSV file
icp_classification_df.to_csv('soil_etl/icp_classification_df.csv', index=False)
print(icp_classification_df)

# -----------------------------------------------------------------------------------------------------------
## 5. Count each type of classification
soil_column = icp_classification_df.columns[1]

def calculate_element_counts(element_column):
    # Group by soil type and count occurrences of each classification, including 'NA' as a string
    element_counts = icp_classification_df.groupby(soil_column)[element_column].value_counts().unstack(fill_value=0)

    # Add missing columns with default value 0 if they do not exist
    for col in ['NA', 'optimum', 'low', 'high']:
        if col not in element_counts.columns:
            element_counts[col] = 0

    # Reorder columns
    element_counts = element_counts[['NA', 'optimum', 'low', 'high']]

    # Reset index to make 'soil' a column
    element_counts.reset_index(inplace=True)

    return element_counts

# List of element columns to process
elements = ['B', 'Mg', 'P', 'S', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn']

# Calculate and print counts for each element
for element in elements:
    element_counts = calculate_element_counts(element)
    print(f"--{element} table--")
    print(element_counts)

# -----------------------------------------------------------------------------------------------------------------
## 6. Bar charts for each element
def plot_element_counts(element_counts, element_name):
    # Set figure size for better readability
    plt.figure(figsize=(12, 6))

    # Number of soil types
    n_soils = len(element_counts)
    # Width of a bar
    bar_width = 0.2

    # Positions of bars on the x-axis
    r1 = np.arange(n_soils)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Plotting the data
    plt.bar(r1, element_counts['NA'], color='b', width=bar_width, edgecolor='gray', label='NA')
    plt.bar(r2, element_counts['optimum'], color='g', width=bar_width, edgecolor='gray', label='optimum')
    plt.bar(r3, element_counts['low'], color='r', width=bar_width, edgecolor='gray', label='low')
    plt.bar(r4, element_counts['high'], color='y', width=bar_width, edgecolor='gray', label='high')

    # Adding labels and title
    plt.xlabel('Soil Type', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(n_soils)], element_counts[soil_column], rotation=45)
    plt.title(f'Frequency of {element_name} Classifications by Soil Type')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# Calculate counts and plot graph for each element
for element in elements:
    element_counts = calculate_element_counts(element)
    plot_element_counts(element_counts, element)

# -----------------------------------------------------------------------------------------------------------------
## 7. krm dataframe

soil_column = 'soil'

# Create a new DataFrame with only the rows for 'krm' soil
krm_soil_df = soil_icp_df[soil_icp_df[soil_column] == 'krm']
columns_to_drop = [0, 1, 2, 3]
krm_soil_df = krm_soil_df.drop(krm_soil_df.columns[columns_to_drop], axis=1)
print(krm_soil_df)

# -----------------------------------------------------------------------------------------------------------------
## 8. Pearson Correlation on krm
corr_matrix = krm_soil_df.corr(method='pearson')
print(corr_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .5})
plt.title('Pearson Correlation Matrix')
plt.show()

# -----------------------------------------------------------------------------------------------------------------
## 9. Factor Analysis on krm

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(krm_soil_df)

# Perform KMO Test
kmo_all, kmo_model = calculate_kmo(krm_soil_df)
print("KMO Test Value:", kmo_model)

# Perform Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(krm_soil_df)
print("Bartlett's Test Chi-Square Value:", chi_square_value)
print("Bartlett's Test p-value:", p_value)

#### BAD RESULT IN THE TEST. THE DATA ARE NOT SUITABLE FOR USING A PRINCIPAL COMPONENT ANALYSIS (PCA) OR A FACTOR ANALYSIS (FA)!!!!!!!!!!!

# Initialize and fit the factor analysis model
n_factors = 10
fa = FactorAnalysis(n_components=n_factors, random_state=0)
X_factor = fa.fit_transform(X)

# The factor loadings (or the 'rotated' data) can be accessed via fa.components_
print("Factor Loadings:\n", fa.components_)  # Each row corresponds to a factor

# Scores of each variable on the factors
factor_scores_df = pd.DataFrame(X_factor, columns=[f'Factor{i+1}' for i in range(n_factors)])
print("Factor Scores:\n", factor_scores_df.head())

# Calculate variance explained by each factor
variance_explained = np.sum(fa.components_**2, axis=1)

# Calculate total variance explained
total_variance = np.sum(variance_explained)

# Calculate the proportion of variance explained
proportion_variance_explained = variance_explained / total_variance

# Print the results
print("Total Variance Explained:", total_variance)
print("Proportion of Variance Explained:", proportion_variance_explained)

# Screeplot
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_factors + 1), variance_explained, 'o-', color='blue')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------------------------------
## 10. PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

# Calculate the square root of eigenvalues (variances)
sqrt_eigenvalues = np.sqrt(pca.explained_variance_)

# Scaling factors for individuals
row_scaling = 1 / sqrt_eigenvalues

# Scaling factors for variables
col_scaling = sqrt_eigenvalues

# ------------------------------------------------------------------------------------------------------------
## 11. HJ-Biplot
plt.figure(figsize=(12, 8))

# Plot individuals
for i in range(principalComponents.shape[0]):
    plt.scatter(principalComponents[i, 0] * row_scaling[0], principalComponents[i, 1] * row_scaling[1], c='r')

# Plot variables
for i in range(X.shape[1]):
    # Arrow start at (0,0), then draw the arrows
    plt.arrow(0, 0, pca.components_[0, i] * col_scaling[0], pca.components_[1, i] * col_scaling[1], color='b', alpha=0.5)
    plt.text(pca.components_[0, i] * col_scaling[0] * 1.2, pca.components_[1, i] * col_scaling[1] * 1.2, krm_soil_df.columns[i], color='g')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('HJ-Biplot')
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------------------------------------
## 12.  krm Boxplot

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create a boxplot for each variable
sns.boxplot(data=krm_soil_df)

# Set title and labels (adjust as necessary)
plt.title('Boxplots of Variables in krm_soil_df')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.show()

# ------------------------------------------------------------------------------------------------------------
## 13. Transform and create a new data frame!!!

# Remove unused 'id' column
soil_icp_df.drop('id', axis=1, inplace=True)
soil_hhxrf_df.drop('id', axis=1, inplace=True)

# Define the columns of interest for zero replacement
columns_to_replace = soil_icp_df.columns[3:]  # All columns after 'rep'

# Zero replacement to ensure it is applied correctly to loaded DataFrames
def replace_zeros_with_real_data(soil_icp_df, soil_hhxrf_df, columns_to_replace):

    # Create a copy of soil_icp_df to work with and keep the original data intact
    soil_mix_df = soil_icp_df.copy()
   
    # Iterate through rows of soil_icp_df
    for idx, row in soil_mix_df.iterrows():

        # Identify columns with zeros
        zero_columns = row[columns_to_replace] == 0
        zero_columns_names = zero_columns[zero_columns].index.tolist()
       
        # Continue only if there are zeros in the current row
        if zero_columns.any():

            # Find a row in soil_hhxrf_df that matches 'soil', 'sample', and 'rep'
            match = soil_hhxrf_df[
                (soil_hhxrf_df['soil'] == row['soil']) &
                (soil_hhxrf_df['sample'] == row['sample']) &
                (soil_hhxrf_df['rep'] == row['rep'])
            ]

            # If there is a corresponding row in soil_hhxrf_df and the values are non-zero, replace them in soil_mix_df
            if not match.empty:
                for column in zero_columns_names:
                    if match.iloc[0][column] != 0:
                        soil_mix_df.at[idx, column] = match.iloc[0][column]
   
    return soil_mix_df

# Call replace function with actual data
soil_mix_df_final = replace_zeros_with_real_data(soil_icp_df, soil_hhxrf_df, columns_to_replace)

# Show the corrected result with the real data
soil_mix_df_final.head()
soil_mix_df_final.to_csv('soil_etl/soil_mix_df_final.csv', index=False)

# --------------------------------------------------------------------------------------------------------------------
## 14. Checking if the new ICP values are in the range
index_column = soil_icp_df.iloc[:, 0]

# Agregamos esta columna a 'soil_mix_df_final'. 
# Puedes cambiar 'nombre_nueva_columna' por el nombre real que quieras darle a la columna.
soil_mix_df_final.insert(0, 'id', index_column)
# Mostramos el dataframe final para verificar
print(soil_mix_df_final)

# Initialize the new DataFrame
icp_classification_mix = ['id'] + list(soil_mix_df_final.columns[1:4]) + nutrients_sandy[1:]
icp_classification_mix_df = pd.DataFrame(columns=icp_classification_mix)

# Iterate over soil_mix_df_final to fill icp_classification_mix_df
for id, row in soil_mix_df_final.iterrows():
    new_id = id + 1
    new_row = [new_id] + list(row[1:4])

    # Determine soil type
    soil_name = row[1]
    soil_type = soil_types_df[soil_types_df['soil'] == soil_name]['type'].iloc[0]

    # Select the appropriate nutrient range table
    nutrient_range_df = sandy_soil_nutrient_range if soil_type == 'sandy' else soil_medium_nutrient_range

    # Compare and categorize values
    for element in nutrients_sandy[1:]:
        element_value = row[element]

        if element_value == 0:  # Check if the value is zero
            category = 'NA'  # Assign None (which will be NULL in CSV)
        else:
            lower_limit = float(nutrient_range_df[nutrient_range_df['limit'] == 'inferior'][element])
            upper_limit = float(nutrient_range_df[nutrient_range_df['limit'] == 'superior'][element])

            if element_value < lower_limit:
                category = 'low'
            elif element_value > upper_limit:
                category = 'high'
            else:
                category = 'optimum'
        
        new_row.append(category)

    icp_classification_mix_df.loc[new_id] = new_row

# Export icp_classification_mix_df to a CSV file
icp_classification_mix_df.to_csv('soil_etl/icp_classification_mix_df.csv', index=False)
print(icp_classification_mix_df)

# -----------------------------------------------------------------------------------------------------------
## 15. Count each type of classification in new dataframe
soil_column = icp_classification_mix_df.columns[1]

def calculate_element_counts(element_column):
    # Group by soil type and count occurrences of each classification, including 'NA' as a string
    element_counts = icp_classification_mix_df.groupby(soil_column)[element_column].value_counts().unstack(fill_value=0)

    # Add missing columns with default value 0 if they do not exist
    for col in ['NA', 'optimum', 'low', 'high']:
        if col not in element_counts.columns:
            element_counts[col] = 0

    # Reorder columns
    element_counts = element_counts[['NA', 'optimum', 'low', 'high']]

    # Reset index to make 'soil' a column
    element_counts.reset_index(inplace=True)

    return element_counts

# List of element columns to process
elements = ['B', 'Mg', 'P', 'S', 'K', 'Ca', 'Mn', 'Fe', 'Cu', 'Zn']

# Calculate and print counts for each element
for element in elements:
    element_counts = calculate_element_counts(element)
    print(f"--{element} table--")
    print(element_counts)

# -----------------------------------------------------------------------------------------------------------------
## 16. Bar charts for each element in new data frame
def plot_element_counts(element_counts, element_name):
    # Set figure size for better readability
    plt.figure(figsize=(12, 6))

    # Number of soil types
    n_soils = len(element_counts)
    # Width of a bar
    bar_width = 0.2

    # Positions of bars on the x-axis
    r1 = np.arange(n_soils)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Plotting the data
    plt.bar(r1, element_counts['NA'], color='b', width=bar_width, edgecolor='gray', label='NA')
    plt.bar(r2, element_counts['optimum'], color='g', width=bar_width, edgecolor='gray', label='optimum')
    plt.bar(r3, element_counts['low'], color='r', width=bar_width, edgecolor='gray', label='low')
    plt.bar(r4, element_counts['high'], color='y', width=bar_width, edgecolor='gray', label='high')

    # Adding labels and title
    plt.xlabel('Soil Type', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(n_soils)], element_counts[soil_column], rotation=45)
    plt.title(f'Frequency of {element_name} Classifications by Soil Type')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# Calculate counts and plot graph for each element
for element in elements:
    element_counts = calculate_element_counts(element)
    plot_element_counts(element_counts, element)