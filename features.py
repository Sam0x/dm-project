import pandas as pd

# Load the dataset with the correct delimiter
file_path = "cardio_train_cleaned.csv"  # Replace with your file path if different
data = pd.read_csv(file_path, sep=';')  # Specify the delimiter as ';'

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

# Save the modified dataset to a new file
output_path = "cardio_train_cleaned_with_features.csv"
data.to_csv(output_path, sep=';', index=False)

print(f"New features added and dataset saved to {output_path}.")
