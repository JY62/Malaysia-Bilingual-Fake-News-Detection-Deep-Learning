# # WordCloud
# import pandas as pd
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import ast
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Combine all tokenized titles and full contexts into a single string
# combined_tokens = " ".join(
#     [" ".join(tokens) for tokens in df['Tokenized_Title']] +
#     [" ".join(tokens) for tokens in df['Tokenized_Full_Context']]
# )
#
# # Generate the word cloud for the combined text
# wordcloud_combined = WordCloud(width=800, height=400, background_color='white').generate(combined_tokens)
#
# # Plot the word cloud for the combined text
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud_combined, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud for Malaysia English Political News')
# plt.show()

# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt
# import ast
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Combine tokenized columns for analysis
# df['Combined_Tokens'] = df['Tokenized_Title'] + df['Tokenized_Full_Context']
#
#
# # Function to generate n-grams
# def get_top_ngrams(corpus, n=None):
#     vec = CountVectorizer(ngram_range=n, tokenizer=lambda x: x, preprocessor=lambda x: x)
#     ngrams = vec.fit_transform(corpus)
#     sum_ngrams = ngrams.sum(axis=0)
#     ngram_freq = [(ngram, sum_ngrams[0, idx]) for ngram, idx in vec.vocabulary_.items()]
#     return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:15]
#
#
# # Separate the data into real and fake news
# real_news = df[df['classification_result'] == 'real']['Combined_Tokens']
# fake_news = df[df['classification_result'] == 'fake']['Combined_Tokens']
#
# # Analyze unigrams, bigrams, and trigrams
# n_grams = {
#     'Unigram': (1, 1),
#     'Bigram': (2, 2),
#     'Trigram': (3, 3)
# }
#
#
# # Function to plot n-grams
# def plot_ngrams(title, genuine_ngrams, deceive_ngrams):
#     # Plot real news n-grams
#     real_labels, real_values = zip(*genuine_ngrams)
#     plt.figure(figsize=(12, 6))
#     plt.barh(real_labels, real_values, color='green')
#     plt.xlabel('Frequency')
#     plt.title(f'Top 15 {title} in Real News')
#     plt.gca().invert_yaxis()
#     plt.show()
#
#     # Plot fake news n-grams
#     fake_labels, fake_values = zip(*deceive_ngrams)
#     plt.figure(figsize=(12, 6))
#     plt.barh(fake_labels, fake_values, color='red')
#     plt.xlabel('Frequency')
#     plt.title(f'Top 15 {title} in Fake News')
#     plt.gca().invert_yaxis()
#     plt.show()
#
#
# # Generate and plot n-grams for each type
# for ngram_type, ngram_range in n_grams.items():
#     real_ngrams = get_top_ngrams(real_news, ngram_range)
#     fake_ngrams = get_top_ngrams(fake_news, ngram_range)
#     plot_ngrams(ngram_type, real_ngrams, fake_ngrams)

# # CountVectorizer
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt
# import numpy as np
# import ast
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Combine tokenized columns for analysis
# df['Combined_Tokens'] = df['Tokenized_Title'] + df['Tokenized_Full_Context']
#
# # Join tokens back into a single string for each row
# df['Text'] = df['Combined_Tokens'].apply(lambda x: ' '.join(x))
#
# # Separate the data into real and fake news
# real_news_text = df[df['classification_result'] == 'real']['Text']
# fake_news_text = df[df['classification_result'] == 'fake']['Text']
#
# # Initialize the CountVectorizer
# vectorizer = CountVectorizer(max_features=30)
#
# # Transform the text data into a bag-of-words model
# real_bow = vectorizer.fit_transform(real_news_text)
# fake_bow = vectorizer.fit_transform(fake_news_text)
#
# # Get feature names and sum of occurrences
# real_word_counts = np.asarray(real_bow.sum(axis=0)).flatten()
# fake_word_counts = np.asarray(fake_bow.sum(axis=0)).flatten()
#
# real_feature_names = vectorizer.get_feature_names_out()
# fake_feature_names = vectorizer.get_feature_names_out()
#
# # Create a DataFrame for visualization
# real_word_freq_df = pd.DataFrame({'Word': real_feature_names, 'Frequency': real_word_counts}).sort_values(
#     by='Frequency', ascending=False)
# fake_word_freq_df = pd.DataFrame({'Word': fake_feature_names, 'Frequency': fake_word_counts}).sort_values(
#     by='Frequency', ascending=False)
#
# # Plot real news word frequencies
# plt.figure(figsize=(12, 6))
# plt.barh(real_word_freq_df['Word'], real_word_freq_df['Frequency'], color='green')
# plt.xlabel('Frequency')
# plt.title('Top 30 Words in Real News')
# plt.gca().invert_yaxis()
# plt.show()
#
# # Plot fake news word frequencies
# plt.figure(figsize=(12, 6))
# plt.barh(fake_word_freq_df['Word'], fake_word_freq_df['Frequency'], color='red')
# plt.xlabel('Frequency')
# plt.title('Top 30 Words in Fake News')
# plt.gca().invert_yaxis()
# plt.show()

# # Pie chart of real and fake news
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
# # Count the occurrences of each label (REAL, FAKE)
# label_counts = df['classification_result'].value_counts()
#
# # Calculate the percentages
# percentages = (label_counts / label_counts.sum()) * 100
#
# # Prepare the labels with counts and percentages
# labels = [f'{label} ({count} - {percentage:.1f}%)'
#           for label, count, percentage in zip(label_counts.index, label_counts, percentages)]
#
# # Plot the pie chart
# plt.figure(figsize=(8, 8))
# wedges, texts, autotexts = plt.pie(
#     label_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['green', 'red'], textprops={'fontsize': 10}
# )
# plt.title('Distribution of Real and Fake News in English')
# plt.axis('equal')
# plt.show()


# # Noise Detection (Low information words)
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import ast
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Combine tokenized columns for analysis
# df['Combined_Tokens'] = df['Tokenized_Title'] + df['Tokenized_Full_Context']
# df['Combined_Text'] = df['Combined_Tokens'].apply(lambda x: " ".join(x))
#
# # Separate the data into real and fake news
# real_news = df[df['classification_result'] == 'real']['Combined_Text']
# fake_news = df[df['classification_result'] == 'fake']['Combined_Text']
#
#
# # Function to calculate low-information words using TF-IDF
# def get_low_information_words(corpus):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(corpus)
#     # Calculate the mean TF-IDF score for each word
#     mean_tfidf_scores = tfidf_matrix.mean(axis=0).A1
#     # Create a dictionary of words with their corresponding mean TF-IDF scores
#     words_scores = {word: mean_tfidf_scores[idx] for word, idx in vectorizer.vocabulary_.items()}
#     # Sort the words by their scores (ascending to get low information words)
#     low_info_words = sorted(words_scores.items(), key=lambda x: x[1])[:50]  # Select top 50 low-information words
#     return dict(low_info_words)
#
#
# # Generate low-information words for real and fake news
# low_info_real = get_low_information_words(real_news)
# low_info_fake = get_low_information_words(fake_news)
#
#
# # Function to create a word cloud from a frequency dictionary
# def plot_wordcloud(freq_dict, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(title)
#     plt.show()
#
#
# # Plot word clouds for low-information words
# plot_wordcloud(low_info_real, "Low-Information Words in Real News")
# plot_wordcloud(low_info_fake, "Low-Information Words in Fake News")

# # Average sentence length
# import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Calculate sentence lengths
# df['Title_Length'] = df['Tokenized_Title'].apply(len)
# df['Full_Context_Length'] = df['Tokenized_Full_Context'].apply(len)
#
# # Separate the data for real and fake news
# real_news = df[df['classification_result'] == 'real']
# fake_news = df[df['classification_result'] == 'fake']
#
# # Prepare data for title visualization
# title_data = pd.DataFrame({
#     'Length': df['Title_Length'].tolist() +
#               real_news['Title_Length'].tolist() +
#               fake_news['Title_Length'].tolist(),
#     'Category': (['Overall'] * len(df) +
#                  ['Real'] * len(real_news) +
#                  ['Fake'] * len(fake_news))
# })
#
# # Prepare data for full context visualization
# full_context_data = pd.DataFrame({
#     'Length': df['Full_Context_Length'].tolist() +
#               real_news['Full_Context_Length'].tolist() +
#               fake_news['Full_Context_Length'].tolist(),
#     'Category': (['Overall'] * len(df) +
#                  ['Real'] * len(real_news) +
#                  ['Fake'] * len(fake_news))
# })
#
# # Plot box plot for titles
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Category', y='Length', data=title_data)
# plt.xticks(rotation=45)
# plt.title('Sentence Length Distribution for EN News Titles')
# plt.ylabel('Number of Tokens')
# plt.xlabel('Category')
# plt.show()
#
# # Plot box plot for full contexts
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Category', y='Length', data=full_context_data)
# plt.xticks(rotation=45)
# plt.title('Sentence Length Distribution for EN News Full Contexts')
# plt.ylabel('Number of Tokens')
# plt.xlabel('Category')
# plt.show()

# # Average Word Length
# import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the dataset
# df = pd.read_csv('Log/Processed_Dataset_EN.csv')
#
#
# # Function to convert string representation of list to list
# def str_to_list(s):
#     return ast.literal_eval(s)
#
#
# # Convert the tokenized columns from strings to lists
# df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
# df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)
#
# # Combine the tokenized columns for analysis
# df['Combined_Tokens'] = df['Tokenized_Title'] + df['Tokenized_Full_Context']
#
#
# # Function to calculate average word length
# def average_word_length(tokens):
#     if len(tokens) == 0:
#         return 0
#     return sum(len(word) for word in tokens) / len(tokens)
#
#
# # Calculate average word length for the combined tokens
# df['Avg_Word_Length'] = df['Combined_Tokens'].apply(average_word_length)
#
# # Separate the data for real and fake news
# real_news = df[df['classification_result'] == 'real']
# fake_news = df[df['classification_result'] == 'fake']
#
# # Prepare data for visualization
# word_length_data = pd.DataFrame({
#     'Avg_Word_Length': df['Avg_Word_Length'].tolist() +
#                        real_news['Avg_Word_Length'].tolist() +
#                        fake_news['Avg_Word_Length'].tolist(),
#     'Category': (['Overall'] * len(df) +
#                  ['Real'] * len(real_news) +
#                  ['Fake'] * len(fake_news))
# })
#
# # Plot box plot for average word length
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Category', y='Avg_Word_Length', data=word_length_data)
# plt.title('Average Word Length Distribution in EN Dataset')
# plt.ylabel('Average Word Length')
# plt.xlabel('Category')
# plt.show()

# Count Unique Words
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Log/Processed_Dataset_EN.csv')


# Function to convert string representation of list to list
def str_to_list(s):
    return ast.literal_eval(s)


# Convert the tokenized columns from strings to lists
df['Tokenized_Title'] = df['Tokenized_Title'].apply(str_to_list)
df['Tokenized_Full_Context'] = df['Tokenized_Full_Context'].apply(str_to_list)

# Combine the tokenized columns for analysis
df['Combined_Tokens'] = df['Tokenized_Title'] + df['Tokenized_Full_Context']

# Calculate unique words for each category
overall_unique_words = set([word for tokens in df['Combined_Tokens'] for word in tokens])
real_unique_words = set([word for tokens in df[df['classification_result'] == 'real']['Combined_Tokens'] for word in tokens])
fake_unique_words = set([word for tokens in df[df['classification_result'] == 'fake']['Combined_Tokens'] for word in tokens])

# Count the number of unique words in each category
unique_word_counts = {
    'Overall': len(overall_unique_words),
    'Real': len(real_unique_words),
    'Fake': len(fake_unique_words)
}

# Convert dict keys and values to lists for plotting
categories = list(unique_word_counts.keys())
counts = list(unique_word_counts.values())

# Create a bar chart for unique word counts
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['blue', 'green', 'red'])
plt.title('Total Number of Unique Words in EN Dataset')
plt.ylabel('Number of Unique Words')
plt.xlabel('Category')
plt.show()
