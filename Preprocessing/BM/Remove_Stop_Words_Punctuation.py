import pandas as pd
import json
import string
from tqdm import tqdm

# Load the CSV dataset
csv_file_path = 'Log/05_Tokenized_BM_News.csv'
df = pd.read_csv(csv_file_path)


def remove_stopwords_punctuation_escape(tokens):
    # Load the Malay stopwords from the JSON file
    with open('stopwords-ms.json', 'r', encoding='utf-8') as f:
        malay_stopwords = set(json.load(f))
    # Define a set of escape characters
    escape_chars = {'\n', '\t', '\r', '\b', '\f', '\\'}
    # Remove stop words, punctuation, and escape characters from tokenized list
    filtered_tokens = [word for word in tokens if word.lower() not in malay_stopwords
                       and word not in string.punctuation
                       and word not in escape_chars
                       and word != "''" and word != '""']
    return filtered_tokens


# Convert the tokenized strings to lists
df['Tokenized_Title'] = df['Tokenized_Title'].apply(eval)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(eval)
# Use tqdm progress bar
tqdm.pandas(desc="Processing")
# Apply the removal function
df['Tokenized_Title'] = df['Tokenized_Title'].progress_apply(remove_stopwords_punctuation_escape)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].progress_apply(remove_stopwords_punctuation_escape)
# Save the modified DataFrame back to a CSV file
df.to_csv('06_NoSW_Puct_Esc_BM_News.csv', index=False)
print("Stop words, punctuation, and escape characters removed successfully.")

