# Inspect the housing data to check for missing values and develop an approach to handle them.#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


# View a sample of the rows with missing values:
print(df[df.isnull().any(axis=1)].head())

# Visualise a sample of the data
data_sample = df.sample(500)
pairplot = sns.pairplot(data_sample, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 10})
pairplot.figure.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
for ax in pairplot.axes[:, 0]:
    ax.yaxis.set_tick_params(rotation=0)
plt.show()
