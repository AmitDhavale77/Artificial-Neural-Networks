# Inspect the housing data to check for missing values and develop an approach to handle them.#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
import torch

# formatting settings
pd.set_option("display.max_columns", None)
sns.set_style(style="whitegrid")

# Load the data
df = pd.read_csv("housing.csv")

# Get the data types of each feature
print(df.dtypes)

# Describe the data
print(df.describe())

# Check for missing values
missing_values = df.isnull().mean().round(decimals=4) * 100
print(f"Percentage of missing values for each feature:")
print(missing_values)

null_values = df.isnull().sum()
print(null_values)

num_rows = df.shape[0]
print("num rows", num_rows)

print(df.head())

lb = LabelBinarizer()

categorical_cols = df.select_dtypes(include=['object']).columns

print(categorical_cols)


binary_cols = lb.fit_transform(df[categorical_cols])

print(binary_cols.shape)

if binary_cols.shape[1] > 1:
    for i in range(binary_cols.shape[1]):
        df[f"{list(categorical_cols)}_{i}"] = binary_cols[:, i]
else:
    df[categorical_cols] = binary_cols

print(df.columns)

df.drop(columns=["Index(['ocean_proximity'], dtype='object')_0"], errors='ignore', inplace=True)

print(df.head())

df.drop(columns=["ocean_proximity"], inplace=True)
# Step 3: Compute the Spearman correlation matrix
spearman_corr = df.corr(method='spearman')
print(spearman_corr)

#Step 5: Create a mask to identify highly correlated features
threshold = 0.8  # Set the threshold for high correlation
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))  # Upper triangle mask to ignore duplicate pairs

# Step 6: Find pairs of features with correlation above the threshold
redundant_features = [
    (column, row)
    for column in spearman_corr.columns
    for row in spearman_corr.columns
    if (column != row) and (spearman_corr.loc[column, row] > threshold) and not mask[spearman_corr.columns.get_loc(column), spearman_corr.columns.get_loc(row)]
]

# Print the redundant feature pairs
print("Highly correlated feature pairs:")
for pair in redundant_features:
    print(pair)

# Step 2: Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    spearman_corr, 
    annot=True,  # Show the correlation coefficients on the heatmap
    cmap='coolwarm',  # Color scheme for positive/negative correlations
    vmin=-1, vmax=1,  # Value range
    center=0, 
    linewidths=0.5, 
    cbar_kws={'shrink': 0.8}
)

# Step 3: Add title and labels
plt.title('Spearman Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

columns_to_drop = ['households', 'total_bedrooms']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

spearman_corr = df.corr(method='spearman')

# Step 3: Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    spearman_corr,
    annot=True,          # Show the correlation coefficients on the heatmap
    cmap='coolwarm',     # Color scheme for positive/negative correlations
    vmin=-1, vmax=1,     # Value range
    center=0,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)

# Step 4: Add title and labels
plt.title('Spearman Correlation Matrix After Dropping Columns', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 1: Define the target variable
target_column = 'median_house_value'  # Update this if your target variable has a different name
ocean_column = "['ocean_proximity']_4"
# Step 2: Split the dataset into X (features) and y (target)
X = df.drop(columns=[target_column, ocean_column], inplace=False)  # Features
y = df[target_column]  # Target variable

# Step 3: Display the shapes of X and y to confirm the split
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print(X.head())
print(df.shape)

print(X.columns)
print(df.columns)

print(y.isnull().sum())

from sklearn.preprocessing import StandardScaler

# Step 1: Identify continuous columns (exclude the one-hot encoded columns)
# Assuming you know which columns are one-hot encoded. Otherwise, identify the categorical ones
one_hot_columns = [col for col in X.columns if '_0' in col or '_1' in col or '_2' in col or '_3' in col]  # Adjust this condition based on your one-hot column names
continuous_columns = [col for col in X.columns if col not in one_hot_columns]

# Step 2: Apply Z-score normalization (standardization) to continuous columns
scaler = StandardScaler()
X[continuous_columns] = scaler.fit_transform(X[continuous_columns])

# Step 3: Check the result
print(f"Columns after Z-score normalization:\n{X.head()}")

X_tensor = torch.tensor(X.values, dtype=torch.float32)  # Convert X to tensor of float type
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Convert y to tensor of float type (or long for classification)


print(X_tensor.shape)
print(y_tensor.shape)








# View a sample of the rows with missing values:
print(df[df.isnull().any(axis=1)].head())

# # Visualise a sample of the data
# data_sample = df.sample(500)
# pairplot = sns.pairplot(data_sample, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 10})
# pairplot.figure.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
# for ax in pairplot.axes[:, 0]:
#     ax.yaxis.set_tick_params(rotation=0)
# plt.show()
