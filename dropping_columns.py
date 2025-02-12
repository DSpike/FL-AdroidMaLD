# %%
import pandas as pd
import os

# Define the path to the directory containing the CSV files
directory = r'C:/Users/USER/Documents/NTUST/Conference_Workshop_Seminar/Android/Dataset/AndMal2020-dynamic-BeforeAndAfterReboot/Cleaned_Files'

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Drop the "Hash" column if it exists
        if 'Hash' in df.columns:
            df.drop(columns=['Hash'], inplace=True)
        
        # Save the modified DataFrame back to CSV
        df.to_csv(file_path, index=False)

print("Hash column dropped and files saved successfully.")


