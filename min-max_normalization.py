# %%
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the input and output directories
input_folder = r'C:/Users/USER/Documents/NTUST/Conference_Workshop_Seminar/Android/Dataset/AndMal2020-dynamic-BeforeAndAfterReboot/Cleaned_Files'
output_folder = os.path.join(input_folder, 'normalized_dataset')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Process each CSV file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Select numeric columns excluding 'Category' and 'Family'
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Category', 'Family'])
        
        # Normalize the data
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Save the normalized data to the output folder
        output_file_path = os.path.join(output_folder, filename)
        df.to_csv(output_file_path, index=False)

print("Normalization complete using MinMaxScaler. Normalized files saved in 'normalized_dataset' folder.")


