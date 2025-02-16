import pandas as pd
import numpy as np

df = pd.read_csv('input_data.csv')

mappings = {
    'Gender': {'Female': 1, 'Male': 0},
    'Work Type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Never Worked': 3, 'Never_worked': 4},
    'Residence Type': {'Urban': 0, 'Rural': 1},
    'Smoking Status': {'Never smoked': 0, 'Formerly smoked': 1, 'Smokes': 2, 'Unknown': 3},
    'Physical Activity': {'Sedentary': 1, 'Active': 0, 'Light': 2, 'Moderate': 3},
    'Dietary Habits': {'Non Vegetarian': 1, 'Vegetarian': 0, 'Mixed': 2},
    'Education Level': {'No Education': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3},
    'Region': {'North': 0, 'South': 1, 'East': 2, 'West': 3},
    'Income Level': {'Low': 0, 'Middle': 1, 'High': 2}
}

for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

df = df.drop(columns=['ID'], errors='ignore')

numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.fillna(numeric_df.median())

mean = numeric_df.mean().round(4)
median = numeric_df.median().round(4)
mode = numeric_df.mode().iloc[0].round(4) if not numeric_df.mode().empty else np.nan
variance = numeric_df.var().round(4)
std_dev = numeric_df.std().round(4)
Q1 = numeric_df.quantile(0.25).round(4)
Q3 = numeric_df.quantile(0.75).round(4)
IQR = (Q3 - Q1).round(4)

results = pd.DataFrame({
    'Measure': ['Mean', 'Median', 'Mode', 'Q1', 'Q3', 'IQR', 'Variance', 'Standard Deviation'],
    **{col: [mean[col], median[col], mode[col], Q1[col], Q3[col], IQR[col], variance[col], std_dev[col]] for col in numeric_df.columns}
})

results.set_index('Measure', inplace=True)
results.to_csv('stroke_prediction_stats.csv')
print("Results saved to 'stroke_prediction_stats.csv'")