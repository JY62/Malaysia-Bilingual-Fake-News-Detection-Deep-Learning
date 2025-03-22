import pandas as pd
from googletrans import Translator

# Load the CSV file
file_path = '/mnt/data/Cleaned_Integrated_Political_EN_News.csv'
df = pd.read_csv(file_path)

# Initialize the translator
translator = Translator()


# Function to translate text
def translate_text(text):
    try:
        # Translate the text to English
        translated = translator.translate(text, src='ms', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# Translate the 'Title' and 'Full_Context' columns
df['Title'] = df['Title'].apply(translate_text)
df['Full_Context'] = df['Full_Context'].apply(translate_text)

# Save the translated DataFrame to a new CSV file
translated_file_path = '/mnt/data/Translated_Political_EN_News.csv'
df.to_csv(translated_file_path, index=False)

print(f"Translation completed and saved to {translated_file_path}.")
