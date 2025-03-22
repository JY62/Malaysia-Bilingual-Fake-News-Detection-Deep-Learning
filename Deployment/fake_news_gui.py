import malaya, torch, pickle, nltk, spacy, string, re, html, json
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from lingua import Language, LanguageDetectorBuilder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
import streamlit as st
import time

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger_eng')

# Initialize Language Detector
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.MALAY).build()

# Initialize NLP tools
nlp_en = spacy.load("en_core_web_sm")
malay_tokenizer = malaya.tokenizer.Tokenizer()
malay_model = malaya.spelling_correction.probability.load()
spell = SpellChecker()
english_lemmatizer = WordNetLemmatizer()
stemmer_malay = malaya.stem.sastrawi()

# File paths for Bag of Words
bow_files = {
    "en_real": "en_real_bow.pkl",
    "en_fake": "en_fake_bow.pkl",
    "bm_real": "bm_real_bow.pkl",
    "bm_fake": "bm_fake_bow.pkl"
}


# Function to load Bag of Words
def load_bag_of_words(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Function to calculate word occurrences based on Bag of Words
def calculate_word_occurrences(tokens, classification, language):
    # Construct the key based on language and classification (e.g., 'English_real', 'Malay_fake')
    key = f"{language.lower()}_{classification.lower()}"
    # Mapping of the keys to the actual file names
    bow_files = {
        "english_real": "en_real_bow.pkl",
        "english_fake": "en_fake_bow.pkl",
        "malay_real": "bm_real_bow.pkl",
        "malay_fake": "bm_fake_bow.pkl"
    }
    # Get the file path for the corresponding Bag of Words file
    file_path = bow_files.get(key)
    if not file_path:
        st.error(f"No Bag of Words available for {key}")
        return pd.Series(dtype='float64')
    # Load the Bag of Words file and count the token occurrences
    word_counts = load_bag_of_words(file_path)
    token_counts = {token: word_counts.get(token, 0) for token in tokens}
    # Return token counts sorted in descending order
    return pd.Series(token_counts).sort_values(ascending=False)


# Function to load and preprocess the dataset
def load_and_preprocess_dataset(path, language):
    dataset = pd.read_csv(path)
    dataset['tokens'] = (
            dataset['Tokenized_Title'].apply(lambda x: eval(x) if isinstance(x, str) else []) +
            dataset['Tokenized_Full_Context'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    )
    dataset['language'] = language
    return dataset


# Function to visualize word occurrences
def visualize_word_occurrences(word_counts, title="Word Occurrences"):
    try:
        plt.figure(figsize=(8, 4))
        word_counts.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel("Words")
        plt.ylabel("Occurrences")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Failed to render visualization: {str(e)}")
        print(f"Visualization error: {e}")


def f1_m(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# Preprocessing Functions
def remove_special_chars_html(text):
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"]+", "", text)
    return text.strip()


def segment_text(text, chunk_size=256):
    """Segments text using a HuggingFace Malaya model."""
    model = malaya.segmentation.huggingface()
    tokens = text.split()
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return ' '.join(model.generate([' '.join(chunk)])[0] for chunk in chunks)


def tokenize_text(text):
    detected_language = detector.detect_language_of(text)
    if detected_language == Language.ENGLISH:
        return [token.text for token in nlp_en(text)]
    elif detected_language == Language.MALAY:
        return malay_tokenizer.tokenize(text)
    else:
        raise ValueError("Unsupported language detected or unable to determine language.")


def remove_stopwords_punctuation(tokens, language):
    escape_chars = {'\n', '\t', '\r', '\b', '\f', '\\'}
    if language == Language.ENGLISH:
        stop_words = set(stopwords.words('english'))
    elif language == Language.MALAY:
        with open('../Dataset_Resource/stopwords-ms.json', 'r', encoding='utf-8') as f:
            stop_words = set(json.load(f))
    else:
        raise ValueError("Unsupported language detected or unable to determine language.")
    return [
        word for word in tokens
        if word.lower() not in stop_words
           and word not in string.punctuation
           and word not in escape_chars
    ]


def correct_spelling(tokens):
    corrected_tokens = []
    for token in tokens:
        detected_language = detector.detect_language_of(token)
        if detected_language == Language.ENGLISH:
            corrected_tokens.append(spell.correction(token))
        elif detected_language == Language.MALAY:
            corrected_tokens.append(malay_model.correct(token))
        else:
            corrected_tokens.append(token)
    return corrected_tokens


def lemmatize_tokens(tokens):
    lemmatized_tokens = []
    for token in tokens:
        if not token or not isinstance(token, str):
            continue
        try:
            detected_language = detector.detect_language_of(token)
        except Exception as e:
            print(f"Error detecting language for token '{token}': {e}")
            continue
        if detected_language == Language.ENGLISH:
            pos_tag = nltk.pos_tag([token])[0][1]
            wordnet_pos = get_wordnet_pos(pos_tag)
            if wordnet_pos:
                lemmatized_tokens.append(english_lemmatizer.lemmatize(token, pos=wordnet_pos))
            else:
                lemmatized_tokens.append(english_lemmatizer.lemmatize(token))
        elif detected_language == Language.MALAY:
            lemmatized_tokens.append(stemmer_malay.stem(token))
        else:
            lemmatized_tokens.append(token)
    return lemmatized_tokens


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None


# Preprocess text for tokens
def preprocess_text(text):
    if pd.isna(text) or not text.strip():
        return []
    print("Original text: ", text)
    text = remove_special_chars_html(text)
    print("Text after removing special chars: ", text)
    # text = segment_text(text)
    # print("Text after segmentation: ", text)
    tokens = tokenize_text(text)
    print("Tokens after tokenization:", tokens)
    tokens = remove_stopwords_punctuation(tokens, detector.detect_language_of(text))
    print("Tokens after removing stopwords:", tokens)
    tokens = correct_spelling(tokens)
    print("Tokens after spelling correction:", tokens)
    tokens = lemmatize_tokens(tokens)
    print("Tokens after lemmatization:", tokens)
    return [token.lower() for token in tokens]


# Load Models and Tokenizers
en_model_path = "../Trained_Models/en_han_model.h5"
en_tokenizer_path = "../Trained_Models/en_tokenizer.pkl"
bm_model_path = "../Trained_Models/bm_bert_model.h5"
bm_tokenizer_path = "../Trained_Models/bm_tokenizer"
bert_model = malaya.transformer.huggingface(model='mesolitica/bert-base-standard-bahasa-cased')

en_model = load_model(en_model_path, custom_objects={'f1_m': f1_m})
with open(en_tokenizer_path, 'rb') as f:
    en_tokenizer = pickle.load(f)
bm_model = load_model(bm_model_path, custom_objects={'f1_m': f1_m})
bm_tokenizer = AutoTokenizer.from_pretrained(bm_tokenizer_path)

MAX_SEQ_LENGTH = 512


# Fake News Classification Function (Integrated)
def classify_fake_news(samples):
    results = []
    for sample in samples:
        title = preprocess_text(sample.get("title", ""))
        context = preprocess_text(sample.get("context", ""))
        full_text = title + context
        language = detector.detect_language_of(' '.join(full_text))
        if language == Language.ENGLISH:
            # Preprocess title and context separately
            title_sequence = en_tokenizer.texts_to_sequences([' '.join(title)])
            context_sequence = en_tokenizer.texts_to_sequences([' '.join(context)])

            padded_title = pad_sequences(title_sequence, maxlen=20, padding='post', truncating='post')
            padded_context = pad_sequences(context_sequence, maxlen=400, padding='post', truncating='post')
            # Predict using HAN model
            probability = float(en_model.predict([padded_title, padded_context])[0][0])
        elif language == Language.MALAY:
            # Concatenate title and context with tagging
            tagged_input = f"[TITLE] {' '.join(title)} [CONTEXT] {' '.join(context)}"
            # Tokenize and chunk the input for BERT
            encoded = bm_tokenizer.encode(tagged_input, add_special_tokens=True, return_tensors="pt")
            chunks = torch.split(encoded, MAX_SEQ_LENGTH - 2, dim=1)
            # Generate embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                decoded_chunk = bm_tokenizer.decode(chunk[0].tolist())  # Decode chunk to text
                chunk_embedding = bert_model.vectorize([decoded_chunk])
                embeddings.append(chunk_embedding)
            # Stack embeddings and reshape to match model input
            embeddings = np.vstack(embeddings)
            embeddings = np.reshape(embeddings, (len(embeddings), -1, 768))
            # Predict probabilities for all chunks
            probabilities = bm_model.predict(embeddings)
            probability = np.mean(probabilities)
        else:
            results.append({"title": title, "context": context, "probability": None,
                            "label": "unsupported language", "language": "None"})
            continue
        label = "real" if probability > 0.5 else "fake"
        results.append({"title": title,
                        "context": context,
                        "probability": probability,
                        "label": label,
                        "language": "English" if language == Language.ENGLISH else "Malay",
                        "tokens": full_text})
    return results


# GUI Implementation
# Streamlit App Initialization
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Malaysia Bilingual Political Fake News Detection System")
st.markdown("""
This application analyzes and classifies political news articles in English and Malay as **Real** or **Fake**. 
It provides additional insights into the model's decision-making process.
""")

# User Input Section
st.header("Enter News Details")
news_title = st.text_area("News Title", placeholder="Type the news title here...", height=100)
news_context = st.text_area("News Context", placeholder="Type the news context here...", height=200)

# Process Button
if st.button("Analyze News"):
    if not news_title.strip() or not news_context.strip():
        st.error("Both News Title and Context are required!")
    else:
        start_time = time.time()
        sample_input = [{"title": news_title, "context": news_context}]
        print("Sample Input: ", sample_input)
        with st.spinner("Processing... This may take a few seconds."):
            predictions = classify_fake_news(sample_input)

        processing_time = time.time() - start_time

        for pred in predictions:
            st.subheader("Analysis Results")
            st.markdown(f"<h4 style='font-size: 18px;'>Detected Language: {pred['language']}</h4>",
                        unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 18px;'>Label: {'Real' if pred['label'] == 'real' else 'Fake'}</h4>",
                        unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 18px;'>Real News Probability: {pred['probability']:.4f}</h4>",
                        unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 18px;'>Processing Time: {processing_time:.4f} seconds</h4>",
                        unsafe_allow_html=True)

            word_occurrences = calculate_word_occurrences(pred['tokens'], pred['label'], pred['language'])
            word_occurrences = word_occurrences.nlargest(8)

            st.subheader(f"Token Contribution Visualization ({pred['language'].capitalize()}, {pred['label'].capitalize()})")
            visualize_word_occurrences(word_occurrences, title=f"Token Contributions for {pred['label'].capitalize()} News")
