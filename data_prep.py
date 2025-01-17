import pandas as pd
from scipy.stats import zscore
from sklearn.impute import KNNImputer

expected_categories = {
    'gender': [1, 2], 
    'cholesterol': [1, 2, 3],
    'gluc': [1, 2, 3],
    'smoke': [0, 1],
    'alco': [0, 1],
    'active': [0, 1],
    'cardio': [0, 1]
}
def get_csv():
    file_path = r'corrupted_cardio_train.csv'
    df = pd.read_csv(file_path)
    return df

def empty_values(df):

    print("Number of missing values in each column:")
    missing_values_per_column = df.isnull().sum()
    print(missing_values_per_column)

    columns_to_drop = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco','active','cardio']
    rows_without_all_data = df.isnull().any(axis=1).sum()
    print(f"Number of rows with empty values: {rows_without_all_data}")
    rows_without_all_data2 = df[columns_to_drop].isnull().any(axis=1).sum()
    print(f"Number of rows with empty categorical values: {rows_without_all_data2}")
    print(f"Number of rows with empty numerical values: {rows_without_all_data-rows_without_all_data2}")

    df = df.dropna(subset=columns_to_drop)
    
    #columns_to_fill = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed

def number_outliers(df):
    # Calculate Z-scores for numerical columns
    columns_to_check = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

    df = check_and_filter_ranges(df)

    # Calculate Z-scores for the specified columns
    z_scores = df[columns_to_check].apply(zscore)
    # Detect rows where any Z-score > 2.5
    outliers = (z_scores.abs() > 3)

    # Print rows with outliers
    print(df[outliers.any(axis=1)])

    #removed outliers
    df_cleaned = df[~outliers.any(axis=1)]
    return df_cleaned

def check_and_filter_ranges(df):
    # Define ranges for each column
    ranges = {
        'height': (125, 200),  # Height in cm
        'weight': (40, 300),  # Weight in kg
        'ap_hi': (40, 400),   # Systolic blood pressure
        'ap_lo': (30, 250)    # Diastolic blood pressure
    }

    for column, (min_val, max_val) in ranges.items():
        if column in df.columns:
            # Identify rows with out-of-range values
            out_of_range = ~df[column].between(min_val, max_val)
            if out_of_range.any():
                print(f"Out-of-range values detected in '{column}':")
                print(df[out_of_range])  # Print rows with out-of-range values
            
            # Remove rows with out-of-range values
            df = df[~out_of_range]
        else:
            print(f"Column '{column}' not found in the DataFrame.")
    
    print(f"Number of rows after filtering out-of-range values: {df.shape[0]}")
    return df

def cat_outliers(df):
    for column, expected in expected_categories.items():
        invalid_rows = df[~df[column].isin(expected)]  # Rows with invalid values
        if not invalid_rows.empty:
            print(f"Row IDs with outliers in {column}:")
            print(invalid_rows['id'])
        else:
            print(f"No outliers detected in {column}.")  
        df_cleaned = df[df[column].isin(expected)]
    return df_cleaned
    
         
    
    

def range_number_columns(df):
    columns_to_check = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    for column in columns_to_check:
        if column in df.columns:
            min_value = df[column].min()
            max_value = df[column].max()
            print(column,"range: [",min_value," - ",max_value,"]")
    
df = get_csv()
print(df.shape)

df_cleaned = empty_values(df)
#int: age, height,ap_hi,ap_lo
#float: weight
#categorical: gender, cholesterol, gluc 
#binary: smoke, alco, active, cardio

print(df_cleaned.shape)
df_cleaned = number_outliers(df_cleaned)
print(df_cleaned.shape)
df_cleaned = cat_outliers(df_cleaned)
print(df_cleaned.shape)
range_number_columns(df_cleaned)

output_path = "cardio_train_cleaned.csv"
df_cleaned.to_csv(output_path, sep=';', index=False)
