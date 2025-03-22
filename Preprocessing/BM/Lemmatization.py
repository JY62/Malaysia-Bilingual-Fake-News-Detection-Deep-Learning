import pandas as pd
import malaya
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm
import ast

# Load the dataset
df = pd.read_csv('Log/07_SpellHandling_BM_News.csv')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the NLTK English lemmatizer
english_lemmatizer = WordNetLemmatizer()

# Initialize the Malaya stemmer
stemmer = malaya.stem.sastrawi()


# Helper function to convert string to list safely
def string_to_list(token_str):
    try:
        return ast.literal_eval(token_str)
    except ValueError:
        # In case the string cannot be converted to a list
        return []


# Preprocess columns to ensure they are lists
df['Tokenized_Title'] = df['Tokenized_Title'].apply(string_to_list)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(string_to_list)


# Function to lemmatize and check language
def lemmatize_and_check_language(token):
    # First try to lemmatize using Malaya and check if it's Malay
    malay_lemmatized = stemmer.stem(token)
    if malaya.dictionary.is_malay(malay_lemmatized, stemmer=stemmer):
        return malay_lemmatized
    else:
        # If not Malay, check if the original token is Malay without lemmatization
        if malaya.dictionary.is_malay(token):
            return token
        # If still not Malay, use NLTK lemmatizer
        return english_lemmatizer.lemmatize(token)


# Function to lemmatize a list of tokens
def lemmatize_tokens(tokens):
    return [lemmatize_and_check_language(token) for token in tokens]


# Apply lemmatization to the specified tokenized columns with progress bar
df['Lemmatized_Title'] = [lemmatize_tokens(tokens) for tokens in tqdm(df['Tokenized_Title'], desc="Lemmatizing Titles")]
df['Lemmatized_Full_Context'] = [lemmatize_tokens(tokens) for tokens in
                                 tqdm(df['Tokenized_Full_Context'], desc="Lemmatizing Full Contexts")]

# Save the updated DataFrame
df.to_csv('08_Lemmatized_BM_News.csv', index=False)
