import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV


def calculate_lift(confusion_matrix):
    # Extract TP, FP, TN, FN from the confusion matrix
    TP = confusion_matrix[1, 1]  # True Positives (class 1 predicted correctly)
    FP = confusion_matrix[0, 1]  # False Positives (class 0 incorrectly predicted as class 1)
    TN = confusion_matrix[0, 0]  # True Negatives (class 0 predicted correctly)
    FN = confusion_matrix[1, 0]  # False Negatives (class 1 incorrectly predicted as class 0) 

    # Model's Precision
    precision_model = TP / (TP + FP)

    # Baseline precision (predicting the majority class)
    precision_baseline = TN / (TN + FN)

    # Lift Calculation
    lift = round(precision_model*100 / precision_baseline,2)
    
    return lift

def print_results(y_test, y_pred, random_search):

    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Lift:', calculate_lift(con_matrix))
    print("\nClassification Report Logistic Regression:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

def logistic_regression_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column,'id'])
    y = df[target_column]
    
    
    param_distributions = {
        'max_iter': np.arange(20, 500, 10)
    }

    model = LogisticRegression()
    random_search = RandomizedSearchCV(estimator=model,param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)
    # Evaluate the model
    


def naive_bayes_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column,'id'])
    y = df[target_column]
    
    
    # Define the Naive Bayes model
    model = GaussianNB()
    random_search = RandomizedSearchCV(estimator=model, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)


def random_forest_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column,'id'])
    y = df[target_column]

    # Define the Random Forest model
    param_distributions = {
        'n_estimators': np.arange(5, 50, 5), 
        'max_depth': np.arange(1, 40, 2),
        'min_samples_leaf': np.arange(1,20,1),
        'min_samples_split': np.arange(1,40)
    }
    model = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)
    
    

def gradient_boosted_trees_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column,'id'])
    y = df[target_column]
    
    
    
    # Define the Gradient Boosted Trees model
    param_distributions = {
        'n_estimators': np.arange(50, 301, 20),  # 50 to 300 with a step of 20
        'max_depth': np.arange(2, 15, 1),       # 2 to 20 with a step of 1
        'learning_rate': np.arange(0.1, 10.1, 1)  # 16 to 256 with a step of 16
    }
    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)


def neural_network_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column, 'id'])
    y = df[target_column]
    
    # Define the Neural Network model
    param_distributions = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],  # Variations of the hidden layer sizes
        'max_iter': [200, 300, 500],  # Number of iterations for training
        'learning_rate_init': np.linspace(0.0001, 0.01, 10),  # Initial learning rate values
    }
    
    
    model = MLPClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)
    
    

def svm_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column, 'id'])
    y = df[target_column]
    
    
    # Define the SVM model    
    param_distributions = {
        'C': np.arange(0.01, 50.1, 0.1),
        'gamma': np.arange(0.01, 50.1, 0.1),
        'kernel': ['rbf']
    }

    model = SVC()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=2, scoring='accuracy', random_state=42, n_jobs=-1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)


def knn_model(df, target_column, numerical_columns, categorical_columns):
    X = df.drop(columns=[target_column, 'id'])
    y = df[target_column]
    
    # Define the kNN model
    param_dist = {
        'n_neighbors': np.arange(1, 51),  # Searching from 1 to 50 neighbors
        'weights': ['distance'],  
        'metric': ['euclidean', 'minkowski']  # Distance metrics to consider
    }
    
    model = KNeighborsClassifier()
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=8, verbose=2, random_state=42, n_jobs=-1)
    
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    print_results(y_test, y_pred, random_search)
    

file_path = r'selected_features_data.csv'
df = pd.read_csv(file_path, sep=';')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

numerical_columns =['ap_hi','Systolic_Ratio','MAP','BMI','age_y'] 
categorical_columns =['gender','cholesterol','gluc','smoke','alco','active','Lifestyle_Risk','Obesity'] 

df[numerical_columns] = StandardScaler().fit_transform(df[numerical_columns])
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

#print("Logistic Regression Model")
#logistic_regression_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
#print("Naive Bayes Model")
#naive_bayes_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
#print("Random Forest Model")
#random_forest_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
#print("Gradient Boosted Trees Model")
#gradient_boosted_trees_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
#print("Neural Network Model")
#neural_network_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
print("SVM Model")
svm_model(df,'cardio',numerical_columns,categorical_columns)
#print("----------------------------------------------")
#print("kNN Network Model")
#knn_model(df,'cardio',numerical_columns,categorical_columns)