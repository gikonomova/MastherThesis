#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats

# Load the dataset
file_path = '/Users/galyaikonomova/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)


# In[20]:


data.describe()


# In[25]:


for column in data.columns:
    if column not in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Frequency data for column: {column}")

        # Get frequency count
        freq_count = data[column].value_counts()

        # Get frequency percentage
        freq_percentage = data[column].value_counts(normalize=True) * 100

        # Combine both into a DataFrame
        freq_df = pd.DataFrame({
            'Frequency Count': freq_count,
            'Frequency Percentage (%)': freq_percentage
        })

        # Display the result
        print(freq_df)
        print("\n")


# In[27]:


for column in data.columns:
    if column in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Summary data for column: {column}")
        print(f"Mean : {data[column].mean():.2f}")
        print(f"Standard Deviation: {data[column].std():.2f}")


# In[28]:


# Split the data into diabetic and non-diabetic
diabetic = data[data['Diabetes_binary'] == 1]
non_diabetic = data[data['Diabetes_binary'] == 0]


# In[29]:


for column in diabetic.columns:
    if column not in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Frequency data for column: {column}")

        # Get frequency count
        freq_count = diabetic[column].value_counts()

        # Get frequency percentage
        freq_percentage = diabetic[column].value_counts(normalize=True) * 100

        # Combine both into a DataFrame
        freq_df = pd.DataFrame({
            'Frequency Count': freq_count,
            'Frequency Percentage (%)': freq_percentage
        })

        # Display the result
        print(freq_df)
        print("\n")


# In[30]:


for column in diabetic.columns:
    if column in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Summary data for column: {column}")
        print(f"Mean : {diabetic[column].mean():.2f}")
        print(f"Standard Deviation: {diabetic[column].std():.2f}")


# In[31]:


for column in non_diabetic.columns:
    if column not in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Frequency data for column: {column}")

        # Get frequency count
        freq_count = non_diabetic[column].value_counts()

        # Get frequency percentage
        freq_percentage = non_diabetic[column].value_counts(normalize=True) * 100

        # Combine both into a DataFrame
        freq_df = pd.DataFrame({
            'Frequency Count': freq_count,
            'Frequency Percentage (%)': freq_percentage
        })

        # Display the result
        print(freq_df)
        print("\n")


# In[32]:


for column in non_diabetic.columns:
    if column in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Summary data for column: {column}")
        print(f"Mean : {non_diabetic[column].mean():.2f}")
        print(f"Standard Deviation: {non_diabetic[column].std():.2f}")


# In[37]:


import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
for var in data.columns:
    if var in ["BMI", "PhysHlth", "GenHlth", "MentHlth"]:
        print(f"Mann-Whitney U Test for variable: {var}")
    
        # Perform the test between diabetic and non-diabetic groups
        stat, p = mannwhitneyu(diabetic[var], non_diabetic[var])

        print(f"U-statistic: {stat}, p-value: {p}")
        print("\n")
    else:
        print(f"Chi-Square Test for variable: {var}")
    
        # Create a contingency table
        contingency_table = pd.crosstab(data['Diabetes_binary'], data[var])

        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        print(f"Chi-Square: {chi2}, p-value: {p}")
        print("Contingency Table:\n", contingency_table)
        print("\n")


# In[12]:


# Create a DataFrame to store the results
table = pd.DataFrame(columns=['Parameter', 'All (%)', 'Diabetes (%)', 'Non-Diabetes (%)', 'p-Value'])


# In[13]:


# Process categorical variables
for var in categorical_vars:
    all_dist = data[var].value_counts(normalize=True) * 100
    diabetes_dist = diabetic[var].value_counts(normalize=True) * 100
    non_diabetes_dist = non_diabetic[var].value_counts(normalize=True) * 100
    
    # Store results in the table
    table = table.append({
        'Parameter': var,
        'All (%)': f"{all_dist[1]:.2f}%" if 1 in all_dist else "0.00%",
        'Diabetes (%)': f"{diabetes_dist[1]:.2f}%" if 1 in diabetes_dist else "0.00%",
        'Non-Diabetes (%)': f"{non_diabetes_dist[1]:.2f}%" if 1 in non_diabetes_dist else "0.00%",
        'p-Value': calculate_p_value(data[[var]], data['Diabetes_binary'])[0]
    }, ignore_index=True)


# In[14]:


# Process continuous variables
for var in continuous_vars:
    all_mean_std = f"{data[var].mean():.2f} ± {data[var].std():.2f}"
    diabetes_mean_std = f"{diabetic[var].mean():.2f} ± {diabetic[var].std():.2f}"
    non_diabetes_mean_std = f"{non_diabetic[var].mean():.2f} ± {non_diabetic[var].std():.2f}"
    
    # Store results in the table
    table = table.append({
        'Parameter': var,
        'All (%)': all_mean_std,
        'Diabetes (%)': diabetes_mean_std,
        'Non-Diabetes (%)': non_diabetes_mean_std,
        'p-Value': calculate_p_value(data[[var]], data['Diabetes_binary'])[0]
    }, ignore_index=True)


# In[16]:


# Display the table
print(table)


# In[ ]:




