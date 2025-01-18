import pandas as pd
from scipy.stats import kstest, norm, anderson

from scipy.stats import shapiro
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset with the correct delimiter
file_path = "cardio_train_cleaned.csv"  # Replace with your file path if different
data = pd.read_csv(file_path, sep=';')  # Specify the delimiter as ';'

def correlation_matrix(df):
    columns_to_check = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    # Filter the DataFrame to include only the specified columns
    filtered_df = df[columns_to_check]

    # Compute the correlation matrix
    correlation_matrix = filtered_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.show()
    
correlation_matrix(data)

# Adding new features
data['height_m'] = data['height'] / 100.0  # Convert height to meters
data['age_y'] = data['age'] / 365.25  # Convert age to years
data['BMI'] = data['weight'] / (data['height_m'] ** 2)  # Body Mass Index
data['Pulse_Pressure'] = data['ap_hi'] - data['ap_lo']  # Pulse Pressure
data['MAP'] = data['ap_lo'] + (data['Pulse_Pressure'] / 3)  # Mean Arterial Pressure
#data['Chol_Gluc_Ratio'] = data['cholesterol'] / data['gluc']  # Cholesterol/Glucose Ratio
data['Lifestyle_Risk'] = data['smoke'] + data['alco'] + (1 - data['active'])  # Lifestyle Risk
data['Systolic_Ratio'] = data['ap_hi'] / data['height']  # Systolic/Height Ratio
data['Diastolic_Ratio'] = data['ap_lo'] / data['height']  # Diastolic/Height Ratio
data['Hypertension'] = ((data['ap_hi'] > 140) | (data['ap_lo'] > 90)).astype(int)  # Hypertension Indicator
data['Obesity'] = (data['BMI'] > 30).astype(int)  # Obesity Indicator


def check_normality_sw(df, columns_to_check):
    for column in columns_to_check:
        if column in df.columns:
            # Perform the Shapiro-Wilk test
            stat, p_value = shapiro(df[column].sample(n = 150,random_state = 10))  # Shapiro ne dela na velikih datasetih
            
            # Print the results
            print(f"Column: {column}")
            print(f"Shapiro-Wilk Statistic: {stat:.4f}, p-value: {p_value:.4e}")
            
            # Interpret the result
            if p_value > 0.05:
                print("The data in this column is likely normally distributed (fail to reject H0).")
            else:
                print("The data in this column is not normally distributed (reject H0).")
            print("-" * 50)
        else:
            print(f"Column '{column}' not found in the DataFrame.")
            print("-" * 50)
def check_normality_ks(df, columns_to_check):
    for column in columns_to_check:
        if column in df.columns:
            data = df[column].dropna()
            mean, std = data.mean(), data.std()
            stat, p_value = kstest(data, 'norm', args=(mean, std))
            print(f"Column: {column}")
            print(f"Kolmogorov-Smirnov Statistic: {stat:.4f}, p-value: {p_value:.4e}")
            if p_value > 0.05:
                print("The data in this column is likely normally distributed (fail to reject H0).")
            else:
                print("The data in this column is not normally distributed (reject H0).")
            print("-" * 50)
        else:
            print(f"Column '{column}' not found in the DataFrame.")
            print("-" * 50)

def check_normality_anderson(df, columns_to_check):
    for column in columns_to_check:
        if column in df.columns:
            result = anderson(df[column].dropna(), dist='norm')  # Normal distribution
            print(f"Column: {column}")
            print(f"Statistic: {result.statistic:.4f}")
            for i, crit in enumerate(result.critical_values):
                sig_level = result.significance_level[i]
                print(f"Critical value at {sig_level}%: {crit:.4f}")
            if result.statistic < result.critical_values[2]:  # Use 5% significance level
                print("The data in this column is likely normally distributed (fail to reject H0).")
            else:
                print("The data in this column is not normally distributed (reject H0).")
            print("-" * 50)
        else:
            print(f"Column '{column}' not found in the DataFrame.")
            print("-" * 50)
# List of columns to check
columns_to_check = [
    'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 
    'Diastolic_Ratio', 'height_m', 'age_y', 'MAP', 
    'Pulse_Pressure', 'Systolic_Ratio'
]

# Call the function
check_normality_sw(data, columns_to_check)

# Save the modified dataset to a new file
output_path = "cardio_train_cleaned_with_features.csv"
data.to_csv(output_path, sep=';', index=False)

print(f"New features added and dataset saved to {output_path}.")
