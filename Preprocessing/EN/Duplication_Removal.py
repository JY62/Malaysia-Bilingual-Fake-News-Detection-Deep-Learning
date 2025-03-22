import pandas as pd

# Load the CSV file
file_path = 'Integrated_Political_EN_News.csv'
df = pd.read_csv(file_path)

# Initial number of rows
initial_rows = len(df)

# Sort the dataframe to prioritize 'FAKE' over 'REAL' in the 'Real_Fake' column
df_sorted = df.sort_values(by='Real_Fake', ascending=False)

# Remove duplicates, keeping the first occurrence (which will be 'FAKE' if there is a conflict)
df_deduced = df_sorted.drop_duplicates(subset=['Title', 'Full_Context'], keep='first')

# Number of rows after deduplication
final_rows = len(df_deduced)

# Calculate the number of duplicates removed
duplicates_removed = initial_rows - final_rows

# Save the deduplicated dataframe to a new CSV file
output_path = 'Integrated_Political_EN_News_deduced.csv'
df_deduced.to_csv(output_path, index=False)

print("Duplicates removed:", duplicates_removed)
print("Deduplicated data saved to:", output_path)


