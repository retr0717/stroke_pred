import pandas as pd
import numpy as np

# Load the dataset (replace 'your_file.csv' with your actual file path)
df = pd.read_csv('Stroke_Prediction_Indians.csv')

# Display the first few rows of the dataset to understand its structure
print("Original Data:")
print(df.head())

# Check the data types of the columns
print("\nData Types Before Conversion:")
print(df.dtypes)

# Convert categorical columns to numeric using appropriate mappings
# Gender: 'Female' = 1, 'Male' = 0
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Work Type: Convert to numeric values based on categories
df['Work Type'] = df['Work Type'].map({
    'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Never Worked': 3, 'Never_worked': 4
})

# Residence Type: 'Urban' = 0, 'Rural' = 1
df['Residence Type'] = df['Residence Type'].map({'Urban': 0, 'Rural': 1})

# Smoking Status: 'Never smoked' = 0, 'Formerly smoked' = 1, 'Smokes' = 2, 'Unknown' = 3
df['Smoking Status'] = df['Smoking Status'].map({'Never smoked': 0, 'Formerly smoked': 1, 'Smokes': 2, 'Unknown': 3})

# Physical Activity: 'Sedentary' = 1, 'Active' = 0, 'Light' = 2, 'Moderate' = 3
df['Physical Activity'] = df['Physical Activity'].map({'Sedentary': 1, 'Active': 0, 'Light': 2, 'Moderate': 3})

# Dietary Habits: 'Non Vegetarian' = 1, 'Vegetarian' = 0, 'Mixed' = 2
df['Dietary Habits'] = df['Dietary Habits'].map({'Non Vegetarian': 1, 'Vegetarian': 0, 'Mixed': 2})

# Education Level: 'No Education' = 0, 'Primary' = 1, 'Secondary' = 2, 'Tertiary' = 3
df['Education Level'] = df['Education Level'].map({'No Education': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3})

# Region: 'North' = 0, 'South' = 1, 'East' = 2, 'West' = 3
df['Region'] = df['Region'].map({'North': 0, 'South': 1, 'East': 2, 'West': 3})

# Income Level: 'Low' = 0, 'Middle' = 1, 'High' = 2
df['Income Level'] = df['Income Level'].map({'Low': 0, 'Middle': 1, 'High': 2})

# Convert 'Stroke Occurrence' to binary (Yes = 1, No = 0)
df['Stroke Occurrence'] = df['Stroke Occurrence'].map({'Yes': 1, 'No': 0})

# Check for any columns that still have NaN after mapping
print("\nColumns with NaN values after mapping:")
print(df.isnull().sum())

# If there are any NaN values, replace them with a default value (e.g., mode or median)
# For example, we could replace missing numeric values with the median of the column
df = df.fillna(df.median())

# Check the data types again after conversion
print("\nData Types After Conversion and Handling Missing Values:")
print(df.dtypes)

# Now, select only numeric columns for the calculations
numeric_df = df.select_dtypes(include=[np.number])

# Print out the summary of the numeric data (ensure there are no NaN values)
print("\nSummary of Numeric Columns:")
print(numeric_df.describe())

# Calculate the measures of center and dispersion

# Mean
mean = numeric_df.mean()
print("\nMean:\n", mean)

# Median
median = numeric_df.median()
print("\nMedian:\n", median)

# Mode (handle the case where mode might be empty)
mode = numeric_df.mode()
if not mode.empty:
    mode_value = mode.iloc[0]  # Take the first mode value (since mode can have multiple values)
else:
    mode_value = np.nan  # If no mode, set as NaN
print("\nMode:\n", mode_value)

# Variance
variance = numeric_df.var()
print("\nVariance:\n", variance)

# Standard Deviation
std_dev = numeric_df.std()
print("\nStandard Deviation:\n", std_dev)

# Calculate the range
data_range = numeric_df.max() - numeric_df.min()
print("\nRange:\n", data_range)

# Interquartile Range (IQR)
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR):\n", IQR)

# Create a new DataFrame to store the results
results = pd.DataFrame({
    'Mean': mean,
    'Median': median,
    'Mode': mode_value,
    'Variance': variance,
    'Standard Deviation': std_dev,
    'Range': data_range,
    'IQR': IQR
})

# Print final results as a summary
print("\nFinal Statistical Results:\n", results)

# Save the results to a new CSV file (or overwrite an existing file)
results.to_csv('stroke_prediction_stats.csv', index=False)

print("\nResults have been saved to 'stroke_prediction_stats.csv'.")