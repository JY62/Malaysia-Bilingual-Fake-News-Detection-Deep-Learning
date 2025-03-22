
import pandas as pd
import ast

# Load the dataset
df = pd.read_csv('Log/09_Standardized_EN_News.csv')  # 'Log/09_Standardized_BM_News.csv'


# Helper function to convert string to list safely
def string_to_list(token_str):
    try:
        return ast.literal_eval(token_str)
    except (ValueError, SyntaxError):
        # In case the string cannot be converted to a list, return an empty list
        return []


# Function to normalize tokens to lowercase
def normalize_tokens(tokens):
    return [token.lower() for token in tokens]


# Convert string representations to lists and normalize to lowercase
df['Tokenized_Title'] = df['Tokenized_Title'].apply(lambda x: normalize_tokens(string_to_list(x)))
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(lambda x: normalize_tokens(string_to_list(x)))

# Save the updated DataFrame
df.to_csv('10_Normalized_EN_News.csv', index=False)  # '10_Normalized_BM_News.csv'

