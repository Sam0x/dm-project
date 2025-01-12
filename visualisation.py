import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset with the correct delimiter
file_path = "cardio_train_with_features.csv"  # Replace with your file path if different
df = pd.read_csv(file_path, sep=';')  # Specify the delimiter as ';'

# Display available features
print("Available features:")
print(df.columns.tolist())

# Ask the user to choose a feature
chosen_feature = input("Enter the feature you want to visualize: ")

# Check if the chosen feature exists in the dataset
if chosen_feature in df.columns:
    # Visualize the distribution
    print(f"Visualizing distribution for: {chosen_feature}")
    
    # Calculate statistics
    mean_value = df[chosen_feature].mean()
    median_value = df[chosen_feature].median()
    mode_value = df[chosen_feature].mode()[0]  # Use the first mode in case of multiple
    
    # Check if the feature is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[chosen_feature]):
        # For numeric features, plot histogram with KDE
        plt.figure(figsize=(10, 6))
        sns.histplot(df[chosen_feature], kde=True, bins=30, color='blue', label='Data')
        
        # Add lines and annotations for mean, median, mode
        plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
        plt.axvline(median_value, color='green', linestyle='-.', label=f'Median: {median_value:.2f}')
        plt.axvline(mode_value, color='purple', linestyle=':', label=f'Mode: {mode_value:.2f}')
        
        plt.title(f"Distribution of {chosen_feature} with Statistics")
        plt.xlabel(chosen_feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    else:
        # For categorical features, plot a count plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=chosen_feature, palette='viridis')
        plt.title(f"Distribution of {chosen_feature}")
        plt.xlabel(chosen_feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
else:
    print(f"Feature '{chosen_feature}' not found in the dataset. Please try again.")
