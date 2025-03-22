# import pandas as pd
#
# # Load the dataset
# file_path = 'Integrated_Political_EN_News_deduced.csv'  # 'Integrated_Political_EN_News_deduced.csv'
# df = pd.read_csv(file_path)
#
# # Count the initial number of rows
# initial_row_count = df.shape[0]
#
# # Remove rows with empty data
# df_cleaned = df.dropna()
#
# # Count the number of rows after cleaning
# final_row_count = df_cleaned.shape[0]
#
# # Calculate the number of rows removed
# rows_removed = initial_row_count - final_row_count
#
# # Print the number of rows removed
# print(f"[{file_path}] Number of rows removed: {rows_removed}")
#
# # Save the cleaned dataset to a new CSV file
# cleaned_file_path = 'Integrated_Political_EN_News_NoNull.csv'  # 'Integrated_Political_EN_News_NoNull.csv'
# df_cleaned.to_csv(cleaned_file_path, index=False)

import pandas as pd
import ast

# Load the CSV file
file_path = 'Log/Testing.csv'
df = pd.read_csv(file_path)


# Define a function to clean tokenized columns
def clean_tokens(token_list_str):
    # Convert string representation of list to an actual list
    token_list = ast.literal_eval(token_list_str)
    # Remove empty strings
    return [token for token in token_list if token]


# Apply the cleaning function to the specified columns
df['Tokenized_Title'] = df['Tokenized_Title'].apply(clean_tokens)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(clean_tokens)

# Save the cleaned DataFrame back to CSV
df.to_csv('Testing.csv', index=False)

print("Tokens with empty strings have been removed.")
