import os
import pandas as pd
from spellchecker import SpellChecker
import malaya
from tqdm import tqdm
import Levenshtein as Lv

# Configure tqdm for pandas integration
tqdm.pandas()

# Set environment variables for Malaya
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the spelling correction model from Malaya for Malay language
malaya_model = malaya.spelling_correction.probability.load()

# Initialize the SpellChecker for English
spell = SpellChecker()


# Function to correct spelling for a token based on detected language
def correct_token(token):
    corrected_token_en = spell.correction(token) if token else token  # Ensure token is not None
    corrected_token_my = malaya_model.correct(token) if token else token
    # Ensure that neither correction is None, fall back to the original token if any correction is None
    if not corrected_token_en:
        corrected_token_en = token
    if not corrected_token_my:
        corrected_token_my = token
    # Calculate edit distances and choose the minimum
    distance_en = Lv.distance(token, corrected_token_en)
    distance_my = Lv.distance(token, corrected_token_my)

    return corrected_token_en if distance_en <= distance_my else corrected_token_my


# Function to correct spelling of tokens in a text
def correct_spelling(tokens):
    corrected_tokens = [correct_token(token) for token in tokens if token]  # Check token is not empty
    return corrected_tokens


# Function to process the tokenized text in DataFrame
def process_text(text):
    if pd.isna(text):
        return ''  # Handle NaN values to avoid further errors
    tokens = text.strip('][').split(', ')  # Adjust parsing if the format differs
    corrected_tokens = correct_spelling(tokens)
    return ','.join(corrected_tokens)


# Load the CSV file
df = pd.read_csv('Log/06_NoSW_Puct_Esc_EN_News.csv')

# Apply spelling correction and display progress
df['Tokenized_Title'] = df['Tokenized_Title'].progress_apply(process_text)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].progress_apply(process_text)

# Save the corrected DataFrame to a new CSV file
df.to_csv('07_SpellHandling_EN_News.csv', index=False)


