import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# File paths for datasets
en_dataset_path = "pages/Processed_Dataset_EN.csv"
bm_dataset_path = "pages/Processed_Dataset_BM.csv"


# Function to load and preprocess dataset
def load_dataset(path):
    dataset = pd.read_csv(path)
    dataset["tokens"] = (
            dataset["Tokenized_Title"].apply(lambda x: eval(x) if isinstance(x, str) else []) +
            dataset["Tokenized_Full_Context"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    )
    dataset["text"] = dataset["tokens"].apply(lambda x: " ".join(x))  # Combine tokens into a single string
    return dataset


# Function to create Bag of Words using CountVectorizer for the specified classification type
def create_bag_of_words(dataset, classification_type):
    print(f"Processing Bag of Words for classification type: {classification_type}")
    filtered_data = dataset[dataset["classification_result"] == classification_type]

    # Use tqdm to monitor the progress of converting text
    texts = []
    for text in tqdm(filtered_data["text"], desc="Processing texts"):
        texts.append(text)

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    word_counts = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.sum(axis=0).A1))  # Convert to dictionary
    return word_counts


# Save Bag of Words to a pickle file
def save_bag_of_words(bag_of_words, filename):
    with open(filename, "wb") as f:
        pickle.dump(bag_of_words, f)
    print(f"Bag of Words saved to {filename}")


# Main workflow
def main():
    # Load datasets with progress monitoring
    print("Loading Dataset")
    en_dataset = load_dataset(en_dataset_path)
    bm_dataset = load_dataset(bm_dataset_path)

    # Create Bag of Words for each category
    print("Creating Bag of Words for en, real")
    en_real_bow = create_bag_of_words(en_dataset, "real")
    save_bag_of_words(en_real_bow, "en_real_bow.pkl")
    print("Creating Bag of Words for en, fake")
    en_fake_bow = create_bag_of_words(en_dataset, "fake")
    save_bag_of_words(en_fake_bow, "en_fake_bow.pkl")
    print("Creating Bag of Words for bm, real")
    bm_real_bow = create_bag_of_words(bm_dataset, "real")
    save_bag_of_words(bm_real_bow, "bm_real_bow.pkl")
    print("Creating Bag of Words for bm, fake")
    bm_fake_bow = create_bag_of_words(bm_dataset, "fake")
    save_bag_of_words(bm_fake_bow, "bm_fake_bow.pkl")


if __name__ == "__main__":
    main()
