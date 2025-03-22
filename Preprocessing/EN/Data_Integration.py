import pandas as pd
import os

# Define the folder containing the CSV files
folder_path = '.'  # '.' refers to the current directory

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Define a standard column order
standard_columns = ['Title', 'Full_Context', 'Source', 'Real_Fake', 'Language']

# Iterate over each CSV file and read it into a DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    # Ensure columns match the standard order
    df = df.reindex(columns=standard_columns)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Define the output CSV file name
output_file = 'Integrated_Political_EN_News.csv'

# Write the combined DataFrame to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"All CSV files have been combined into {output_file}.")


