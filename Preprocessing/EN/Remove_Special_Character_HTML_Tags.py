import pandas as pd
import re
import html

# File paths
input_file_path = 'Integrated_Political_EN_News_NoNull.csv'  # 'Integrated_Political_EN_News_NoNull.csv'
output_file_path = 'Cleaned_Integrated_Political_EN_News.csv'  # 'Cleaned_Integrated_Political_EN_News.csv'


# Function to remove HTML tags and special characters
def clean_text(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    # Remove special characters except for basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]+', '', text)
    return text


def clean_and_save_csv(input_file_name, output_file_name):
    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Apply the clean_text function to the Title and Full_Context columns
    df['Title'] = df['Title'].apply(clean_text)
    df['Full_Context'] = df['Full_Context'].apply(clean_text)

    # Save the cleaned dataframe to a new CSV file
    df.to_csv(output_file_name, index=False)


# Execute the function
clean_and_save_csv(input_file_path, output_file_path)



