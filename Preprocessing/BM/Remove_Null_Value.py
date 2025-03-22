import pandas as pd

# Load the dataset
file_path = 'Integrated_Political_BM_News_deduced.csv'  # 'Integrated_Political_EN_News_deduced.csv'
df = pd.read_csv(file_path)

# Count the initial number of rows
initial_row_count = df.shape[0]

# Remove rows with empty data
df_cleaned = df.dropna()

# Count the number of rows after cleaning
final_row_count = df_cleaned.shape[0]

# Calculate the number of rows removed
rows_removed = initial_row_count - final_row_count

# Print the number of rows removed
print(f"[{file_path}] Number of rows removed: {rows_removed}")

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'Integrated_Political_BM_News_NoNull.csv'  # 'Integrated_Political_EN_News_NoNull.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

