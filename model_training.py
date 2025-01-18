import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

file_path = r'selected_features_data.csv'
df = pd.read_csv(file_path, sep=';')
print(df.head())

def logistic_regression_model(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column,'id'])
    y = df[target_column]
    
    # Identify numerical and categorical columns
    numerical_columns =[] #todo []#X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns =[] #todo []#X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing: Scaling for numerical, one-hot encoding for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first'), categorical_columns)
        ]
    )
    
    # Define the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return pipeline

