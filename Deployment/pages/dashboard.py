import streamlit as st
import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct absolute paths to the datasets
english_dataset_path = os.path.join(current_dir, 'Processed_Dataset_EN.csv')
malay_dataset_path = os.path.join(current_dir, 'Processed_Dataset_BM.csv')

# Folder to save charts
chart_save_path = os.path.join(current_dir, "saved_charts")
os.makedirs(chart_save_path, exist_ok=True)

# Load the datasets
english_dataset = pd.read_csv(english_dataset_path)
malay_dataset = pd.read_csv(malay_dataset_path)


# Helper function for n-grams
def get_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


# Generate top n-grams
def get_top_ngrams(data, n, top_k=10):
    ngram_counter = Counter()
    for text in data:
        tokens = ast.literal_eval(text)  # Convert string to list
        ngrams = get_ngrams(tokens, n)
        ngram_counter.update(ngrams)
    return ngram_counter.most_common(top_k)


# Function to compute sentence lengths
def compute_sentence_lengths(data_column):
    return data_column.apply(lambda x: len(ast.literal_eval(x)))


# Function to save a chart
def save_chart(fig, filename):
    filepath = os.path.join(chart_save_path, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


# Function to load a saved chart
def load_chart(filename):
    filepath = os.path.join(chart_save_path, filename)
    if os.path.exists(filepath):
        return Image.open(filepath)
    return None


# Page Title
st.title("Fake News Detection Dashboard")

# Tabs for English and Malay
language_tab = st.radio("Select Dataset Language", ["English", "Malay"])
dataset = english_dataset if language_tab == "English" else malay_dataset

# N-gram Visualization
st.subheader(f"N-gram Analysis of Fake News ({language_tab})")
fake_news = dataset[dataset['classification_result'] == 'fake']

for n in range(1, 4):  # Unigram, Bigram, Trigram
    chart_filename = f"{language_tab}_top_{n}grams.png"
    saved_chart = load_chart(chart_filename)

    if saved_chart:
        st.markdown(f"#### Top {n}-grams")
        st.image(saved_chart, caption=f"Top {n}-grams in Fake News ({language_tab})")
    else:
        st.markdown(f"#### Top {n}-grams")
        top_ngrams = get_top_ngrams(fake_news['Tokenized_Full_Context'], n)
        ngram_df = pd.DataFrame(top_ngrams, columns=['N-gram', 'Count'])
        ngram_df['N-gram'] = ngram_df['N-gram'].apply(lambda x: ' '.join(x))

        # Plot Bar Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Count', y='N-gram', data=ngram_df, ax=ax, palette="viridis")
        ax.set_title(f"Top {n}-grams in Fake News ({language_tab})")
        save_chart(fig, chart_filename)  # Save the chart
        st.pyplot(fig)

# Sentence Length Distribution
st.subheader(f"Sentence Length Distribution ({language_tab})")

# Compute lengths for titles and contexts
real_title_lengths = compute_sentence_lengths(dataset[dataset['classification_result'] == 'real']['Tokenized_Title'])
fake_title_lengths = compute_sentence_lengths(dataset[dataset['classification_result'] == 'fake']['Tokenized_Title'])

real_context_lengths = compute_sentence_lengths(
    dataset[dataset['classification_result'] == 'real']['Tokenized_Full_Context'])
fake_context_lengths = compute_sentence_lengths(
    dataset[dataset['classification_result'] == 'fake']['Tokenized_Full_Context'])

# Combine lengths into a single DataFrame for comparison
title_lengths_df = pd.DataFrame({
    "Length": pd.concat([real_title_lengths, fake_title_lengths], ignore_index=True),
    "Class": ["Real"] * len(real_title_lengths) + ["Fake"] * len(fake_title_lengths)
})

context_lengths_df = pd.DataFrame({
    "Length": pd.concat([real_context_lengths, fake_context_lengths], ignore_index=True),
    "Class": ["Real"] * len(real_context_lengths) + ["Fake"] * len(fake_context_lengths)
})

# Titles Box Plot
title_chart_filename = f"{language_tab}_title_lengths.png"
saved_title_chart = load_chart(title_chart_filename)

if saved_title_chart:
    st.markdown("### Sentence Lengths in Titles")
    st.image(saved_title_chart, caption=f"Title Sentence Lengths Comparison ({language_tab})")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=title_lengths_df, x="Class", y="Length", ax=ax)
    ax.set_title(f"Title Sentence Lengths Comparison ({language_tab})")
    save_chart(fig, title_chart_filename)
    st.pyplot(fig)

# Contexts Box Plot
context_chart_filename = f"{language_tab}_context_lengths.png"
saved_context_chart = load_chart(context_chart_filename)

if saved_context_chart:
    st.markdown("### Sentence Lengths in Full Contexts")
    st.image(saved_context_chart, caption=f"Context Sentence Lengths Comparison ({language_tab})")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=context_lengths_df, x="Class", y="Length", ax=ax)
    ax.set_title(f"Context Sentence Lengths Comparison ({language_tab})")
    save_chart(fig, context_chart_filename)
    st.pyplot(fig)

