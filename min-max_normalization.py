# %%
import os
import pandas as pd

# Define the input and output directories
input_folder = r'C:/Users/USER/Documents/NTUST/Conference_Workshop_Seminar/Android/Dataset/AndMal2020-dynamic-BeforeAndAfterReboot/Cleaned_Files'
output_folder = os.path.join(input_folder, 'normalized_dataset')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to normalize the data
def min_max_normalize(df):
    # Exclude 'Category' and 'Family' columns
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=['float64', 'int64']).columns

    # Apply min-max normalization
    df_normalized[numeric_cols] = (df_normalized[numeric_cols] - df_normalized[numeric_cols].min()) / (df_normalized[numeric_cols].max() - df_normalized[numeric_cols].min())
    
    return df_normalized

# Process each CSV file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Normalize the data
        df_normalized = min_max_normalize(df)

        # Save the normalized data to the output folder
        output_file_path = os.path.join(output_folder, filename)
        df_normalized.to_csv(output_file_path, index=False)

print("Normalization complete. Normalized files saved in 'normalized_dataset' folder.")


