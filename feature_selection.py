import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

# Load the CSV file
file_path = r'C:\Users\urban\Desktop\corrupted_cardio_train.csv'
data = pd.read_csv(file_path)

# Display the first 15 rows and dataset info
print(data.head(15))
print(data.info())

# Exclude the 'id' column if it exists
columns_to_include = data.loc[:, data.columns != 'id']

# Calculate additional statistics
additional_stats = {
    'Median': columns_to_include.median(numeric_only=True),  # Median
    'Var': columns_to_include.var(numeric_only=True),        # Variance
    'Skew': columns_to_include.skew(numeric_only=True),      # Skewness
    'Kurt': columns_to_include.kurt(numeric_only=True),      # Kurtosis
}

# Combine the default and additional statistics
extended_stats = pd.concat(
    [columns_to_include.describe(include='all').T, pd.DataFrame(additional_stats).T],
    axis=1
)

# Display the extended statistics
print(extended_stats)
data['height_m'] = data['height'] / 100.0  # Convert height to meters
data['age_y'] = data['age'] / 365.25  # Convert age from days to years
data['BMI'] = data['weight'] / (data['height_m'] ** 2)  # Calculate Body Mass Index
data['Pulse_Pressure'] = data['ap_hi'] - data['ap_lo']  # Calculate Pulse Pressure
data['MAP'] = data['ap_lo'] + (data['Pulse_Pressure'] / 3)  # Calculate Mean Arterial Pressure
data['Chol_Gluc_Ratio'] = data['cholesterol'] / data['gluc']  # Cholesterol/Glucose Ratio
data['Lifestyle_Risk'] = data['smoke'] + data['alco'] + (1 - data['active'])  # Lifestyle Risk Score
data['Systolic_Ratio'] = data['ap_hi'] / data['height']  # Systolic/Height Ratio
data['Diastolic_Ratio'] = data['ap_lo'] / data['height']  # Diastolic/Height Ratio
data['Hypertension'] = ((data['ap_hi'] > 140) | (data['ap_lo'] > 90)).astype(int)  # Hypertension Indicator
data['Obesity'] = (data['BMI'] > 30).astype(int)  # Obesity Indicator
print(data.head(15))

target_column = 'cardio'  
X = data.drop(columns=[target_column])
y = data[target_column]

# 1. Information Gain (Mutual Information)
info_gain = mutual_info_classif(X, y, random_state=42)

# 2. Information Ratio (Information Gain / Feature Entropy)
def calculate_information_ratio(feature, target):
    mi = mutual_info_score(feature, target)
    h_feature = entropy(feature.value_counts(normalize=True), base=2)
    return mi / h_feature if h_feature > 0 else 0

info_ratio = [calculate_information_ratio(X[col], y) for col in X.columns]

# 3. Chi-Squared Test
chi_scores, p_values = chi2(X, y)

# 4. Correlation
correlations = X.corrwith(y)

# Normalize the metrics to [0, 1]
scaler = MinMaxScaler()
normalized_metrics = pd.DataFrame({
    'Information Gain': scaler.fit_transform(np.array(info_gain).reshape(-1, 1)).flatten(),
    'Information Ratio': scaler.fit_transform(np.array(info_ratio).reshape(-1, 1)).flatten(),
    'Chi-Squared': scaler.fit_transform(np.array(chi_scores).reshape(-1, 1)).flatten(),
    'Correlation': scaler.fit_transform(np.abs(correlations).fillna(0).values.reshape(-1, 1)).flatten()
}, index=X.columns)

# Combine the metrics (average or weighted sum)
normalized_metrics['Combined Score'] = normalized_metrics.mean(axis=1)

# 5. Use Random Forest for Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
normalized_metrics['Random Forest Importance'] = scaler.fit_transform(
    np.array(rf_model.feature_importances_).reshape(-1, 1)
).flatten()

# Final Score (combine with Random Forest weights)
normalized_metrics['Final Score'] = normalized_metrics[['Combined Score', 'Random Forest Importance']].mean(axis=1)

# Assign Inclusion (0 or 1) based on a threshold
threshold = 0.5  # Adjust threshold as needed
normalized_metrics['Include'] = (normalized_metrics['Final Score'] >= threshold).astype(int)

# Display results
print(normalized_metrics.sort_values(by='Final Score', ascending=False))