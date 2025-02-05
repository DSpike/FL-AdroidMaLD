# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory
base_directory = r"C:\Users\Dspike\Documents\NTUST\Dataset\AndMal2020-dynamic-BeforeAndAfterReboot\Cleaned_Files"
output_directory = r"C:\Users\Dspike\Documents\NTUST\Dataset\AndMal2020-dynamic-BeforeAndAfterReboot\Outliers"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List of prefix column names to check for outliers
prefixes = ["API"] #"", "", "Process", "Battery", "Logcat" #API Memory #Network_TotalTransmittedPackets

# Function to identify outliers using IQR
def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Iterate over all CSV files in the directory
for filename in os.listdir(base_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(base_directory, filename)
        data = pd.read_csv(file_path)

        # DataFrame to collect outliers
        all_outliers = pd.DataFrame()

        # Check for specified features
        for prefix in prefixes:
            columns = [col for col in data.columns if col.startswith(prefix)]
            for column in columns:
                outliers = find_outliers_iqr(data, column)
                outliers['Feature'] = column  # Add a column for the feature name
                outliers['File'] = filename    # Add a column for the file name
                all_outliers = pd.concat([all_outliers, outliers], ignore_index=True)

                # Plot the outliers with a box plot
                plt.figure(figsize=(8,6),dpi=300)
                sns.boxplot(x=data[column], color='orange')
                plt.title(f'Box Plot of {column} in {filename}', fontsize=8)
                plt.show()

                # Scatter plot
                plt.figure(figsize=(8, 6 ),dpi=300)
                plt.scatter(data.index, data[column], label='Data Points', color='brown', s=8)
                plt.scatter(outliers.index, outliers[column], color='purple', label='Outliers', marker='*' )
                plt.title(f'Scatter Plot of {column} in {filename}', fontsize=8)
                plt.ylabel('API Call Frequency',fontsize=8)
                plt.xlabel(column, fontsize=8)
                # Set x-axis ticks to be empty
                plt.xticks([])
                plt.legend()
                plt.show()

        # Save outliers to a CSV file
        if not all_outliers.empty:
            output_file_path = os.path.join(output_directory, f'outliers_{filename}')
            all_outliers.to_csv(output_file_path, index=False)
            print(f'Outliers saved to {output_file_path}')


