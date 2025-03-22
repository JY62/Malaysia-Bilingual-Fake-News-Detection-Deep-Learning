import pandas as pd
import malaya

# Load the CSV file, handling potential issues with quotation marks
file_path = 'Log/05_Tokenized_BM_News.csv'
df = pd.read_csv(file_path, quotechar='"', escapechar='\\')

# Initialize the tokenizer from Malaya
tokenizer = malaya.tokenizer.Tokenizer()

# Apply tokenizer on Title and Full_Context columns
df['Tokenized_Title'] = df['Title'].progress_apply(tokenizer.tokenize)
df['Tokenized_Full_Context'] = df['Full_Context'].progress_apply(tokenizer.tokenize)

# Save the updated DataFrame back to CSV
df.to_csv(file_path, index=False)

print("Tokenization complete. The tokenized data has been saved to", file_path)


