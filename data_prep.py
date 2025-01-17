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
    file_path = r'C:\Users\TrpinM\OneDrive - Soudal N.V\Namizje\School\2Letnik\DM\project\corrupted_cardio_train.csv'
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

    # Calculate Z-scores for the specified columns
    z_scores = df[columns_to_check].apply(zscore)
    # Detect rows where any Z-score > 3
    outliers = (z_scores.abs() > 3)

    # Print rows with outliers
    print(df[outliers.any(axis=1)])

    #todo ki nrdit z njimi


def cat_outliers(df):
    for column, expected in expected_categories.items():
        invalid_rows = df[~df[column].isin(expected)]  # Rows with invalid values
        if not invalid_rows.empty:
            print(f"Row IDs with outliers in {column}:")
            print(invalid_rows['id'])
        else:
            print(f"No outliers detected in {column}.")   

def range_number_columns(df):
    columns_to_check = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    for column in columns_to_check:
        if column in df.columns:
            min_value = df[column].min()
            max_value = df[column].max()
            print(column,"range: [",min_value," - ",max_value,"]")
    
df = get_csv()
print(df.shape)

df_full_rows = empty_values(df)
#int: age, height,ap_hi,ap_lo
#float: weight
#categorical: gender, cholesterol, gluc 
#binary: smoke, alco, active, cardio

print(df_full_rows.shape)
number_outliers(df_full_rows)
cat_outliers(df_full_rows)
range_number_columns(df_full_rows)
