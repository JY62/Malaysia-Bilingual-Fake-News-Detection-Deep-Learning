import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import ast

# Load the CSV file
csv_file_path = 'Log/05_Tokenized_EN_News.csv'
df = pd.read_csv(csv_file_path)


def safe_literal_eval(s):
    """Safely evaluate a string representation of a list."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {s}\nException: {e}")
        return []  # Return an empty list or handle it appropriately


def remove_stopwords_and_punctuation(tokens):
    # Download the stopwords if not already downloaded
    nltk.download('stopwords')

    # Define stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    # Define a set of escape characters
    escape_chars = {'\n', '\t', '\r', '\b', '\f', '\\'}
    """Remove stop words, punctuation, and escape characters from a list of tokens."""
    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words and word not in punctuation and word not in escape_chars and word != "''" and word != '""'
    ]
    return filtered_tokens


# Safely convert the tokenized strings to lists
df['Tokenized_Title'] = df['Tokenized_Title'].apply(safe_literal_eval)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(safe_literal_eval)

# Apply the function to the Tokenized_Title and Tokenized_Full_Context columns
df['Tokenized_Title'] = df['Tokenized_Title'].apply(remove_stopwords_and_punctuation)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(remove_stopwords_and_punctuation)

# Save the modified DataFrame back to a CSV file
df.to_csv('06_NoSW_Puct_Esc_EN_News.csv', index=False)

print("Stop words, punctuation, and escape characters removed successfully.")
