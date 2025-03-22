import pandas as pd
import spacy

# Load the CSV file
file_path = 'Tokenized_Political_EN_News.csv'
df = pd.read_csv(file_path)


# Tokenization Function
def tokenize_text(text):
    # Load spaCy English model
    nlp_en = spacy.load("en_core_web_sm")
    text = text.strip()  # Trim whitespace from the text
    doc = nlp_en(text)
    # Extract tokens as a list of strings
    return [token.text for token in doc]


# Apply tokenization to Title and Full_Context columns
df['Tokenized_Title'] = df['Title'].progress_apply(tokenize_text)
df['Tokenized_Full_Context'] = df['Full_Context'].progress_apply(tokenize_text)

# Add new tokenized columns for tokenized items to the CSV file
df.to_csv(file_path, index=False)
print(f"Updated CSV file with tokenized columns saved at: {file_path}")


