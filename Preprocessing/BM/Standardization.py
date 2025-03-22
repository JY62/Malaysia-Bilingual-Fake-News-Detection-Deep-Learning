import pandas as pd

# Load the dataset
df = pd.read_csv('Log/08_Lemmatized_EN_News.csv')  # 'Log/08_Lemmatized_BM_News.csv'


# Function to standardize the Real_Fake column
def standardize_real_fake(value):
    if value.lower() == 'real':
        return 'REAL'
    elif value.lower() == 'fake':
        return 'FAKE'
    else:
        return value  # Return the original value if it doesn't match expected inputs


# Function to standardize the Language column
def standardize_language(value):
    if value == 'EN':
        return 'ENGLISH'
    elif value == 'BM':
        return 'MALAY'
    else:
        return value  # Return the original value if it doesn't match expected inputs


# Apply standardization to the Real_Fake and Language columns
df['Real_Fake'] = df['Real_Fake'].apply(standardize_real_fake)
df['Language'] = df['Language'].apply(standardize_language)

# Save the updated DataFrame
df.to_csv('09_Standardized_EN_News.csv', index=False)  # 'Log/09_Standardized_BM_News.csv'


