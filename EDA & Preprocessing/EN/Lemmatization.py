import pandas as pd
import malaya
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from tqdm import tqdm
import ast

# Load the dataset
df = pd.read_csv('Log/06_NoSW_Puct_Esc_EN_News.csv')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizers
english_lemmatizer = WordNetLemmatizer()
sastrawi = malaya.stem.sastrawi()


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


# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Function to determine if a word is in English
def is_english(word):
    try:
        return bool(wordnet.synsets(word))
    except:
        return False


# Function to lemmatize a single token with POS tagging
def lemmatize_token_with_pos(token, pos):
    if is_english(token):
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos:
            return english_lemmatizer.lemmatize(token, pos=wordnet_pos)
        else:
            return english_lemmatizer.lemmatize(token)  # Default to noun
    else:
        return sastrawi.stem(token)


# Function to lemmatize a list of tokens with POS tagging
def lemmatize_tokens(tokens):
    # Perform POS tagging on the tokens
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatize_token_with_pos(token, pos) for token, pos in pos_tags]


# Apply lemmatization to the tokenized columns with progress bar
df['Tokenized_Title'] = [lemmatize_tokens(tokens) for tokens in tqdm(df['Tokenized_Title'], desc="Lemmatizing Titles")]
df['Tokenized_Full_Context'] = [lemmatize_tokens(tokens) for tokens in
                                tqdm(df['Tokenized_Full_Context'], desc="Lemmatizing Full Contexts")]
# Save the updated DataFrame
df.to_csv('08_Lemmatized_EN_News.csv', index=False)
