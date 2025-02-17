import pandas as pd
import numpy as np

df = pd.read_csv('air_quality.csv')

print("First five rows of dataset:")
print(df.head())
print("\nShape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())

missing_per_column = df.isnull().sum()
print("\nMissing data column-wise:")
print(missing_per_column)

print("\nTotal number of missing values:", missing_per_column.sum())

columns_with_null = df.isnull().any()
print("\nColumns with missing data:")
print(columns_with_null)

# Filling with mean
filled_mean = df.fillna(df.select_dtypes(include=[np.number]).mean())
print("\nStatus after filling with mean (missing values per column):")
print(filled_mean.isnull().sum())

# Filling with median
filled_median = df.fillna(df.select_dtypes(include=[np.number]).median())
print("\nStatus after filling with median (missing values per column):")
print(filled_median.isnull().sum())

# Filling with mode
filled_mode = df.fillna(df.mode().iloc[0])
print("\nStatus after filling with mode (missing values per column):")
print(filled_mode.isnull().sum())

print("\nDataset with missing data (first five rows):")
print(df[df.isnull().any(axis=1)].head())

print("\nStatus of dataset with missing data:")
print(df.isnull().sum())

dropped_na = df.dropna()
print("\nStatus after dropping missing values (missing values per column):")
print(dropped_na.isnull().sum())

# Existing mapping and statistical calculations code goes here after this block.