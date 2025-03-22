import os
import pandas as pd
import malaya

# Set environment variables for Malaya
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the spelling correction model from Malaya
malaya_model = malaya.spelling_correction.probability.load()


# Function to correct spelling using the Malay model
def correct_spelling(tokens):
    corrected_tokens = []
    for token in tokens:
        # Get the Malay correction
        corrected_token = malaya_model.correct(token)
        corrected_tokens.append(corrected_token)
    return corrected_tokens


# Load the CSV file
df = pd.read_csv('Log/06_NoSW_Puct_Esc_BM_News.csv')
# Perform Spelling Correction
df['Tokenized_Title'] = df['Tokenized_Title'].progress_apply(lambda x: ','.join(correct_spelling(x.split(','))))
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].progress_apply(lambda x: ','.join(correct_spelling(x.split(','))))
# Save the corrected DataFrame to a new CSV file
df.to_csv('07_SpellHandling_BM_News.csv', index=False)


