import os
import ast
import pandas as pd
import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D, \
    Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Save and load paths
SAVE_DIR = "./saved_smoteenn_data"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_data(data, labels, prefix):
    """
    Save data and labels to disk for reproducibility.
    """
    np.save(os.path.join(SAVE_DIR, f"{prefix}_data.npy"), data)
    np.save(os.path.join(SAVE_DIR, f"{prefix}_labels.npy"), labels)


def load_data(prefix):
    """
    Load data and labels from disk.
    """
    data = np.load(os.path.join(SAVE_DIR, f"{prefix}_data.npy"))
    labels = np.load(os.path.join(SAVE_DIR, f"{prefix}_labels.npy"))
    return data, labels


# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
df = pd.read_csv(file_path)

# Extract and preprocess input attributes
titles = df['Tokenized_Title'].apply(ast.literal_eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(ast.literal_eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values

# Convert tokenized lists to strings
title_texts = [" ".join(title) for title in titles]
context_texts = [" ".join(context) for context in contexts]

# Concatenate all texts for tokenizer fitting
all_texts = title_texts + context_texts

# Initialize and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)

# Convert text to sequences for numerical representation
title_sequences = tokenizer.texts_to_sequences(title_texts)
context_sequences = tokenizer.texts_to_sequences(context_texts)

# Pad sequences
maxlen_title = 20
maxlen_context = 400
padded_titles = pad_sequences(title_sequences, maxlen=maxlen_title)
padded_contexts = pad_sequences(context_sequences, maxlen=maxlen_context)

# Combine titles and contexts for SMOTEENN
combined_features = np.hstack([padded_titles, padded_contexts])


def apply_smoteenn(X, y):
    """
    Apply SMOTEENN to balance the dataset.
    :param X: Input features (numerical representations like padded sequences).
    :param y: Labels.
    :return: Balanced features and labels.
    """
    smoteenn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)
    return X_resampled, y_resampled


# Apply SMOTEENN and save the data
print('Start SMOTEEN')
if not os.path.exists(os.path.join(SAVE_DIR, "combined_data.npy")):
    combined_features_smoteenn, labels_smoteenn = apply_smoteenn(combined_features, labels)
    save_data(combined_features_smoteenn, labels_smoteenn, "combined")
else:
    combined_features_smoteenn, labels_smoteenn = load_data("combined")

# Print the number of records after SMOTEENN
print(f"Number of records after SMOTEENN: {combined_features_smoteenn.shape[0]}")

# Split back into titles and contexts after SMOTEENN
padded_titles_smoteenn = combined_features_smoteenn[:, :maxlen_title]
padded_contexts_smoteenn = combined_features_smoteenn[:, maxlen_title:]

# Split the SMOTEENN-processed combined data
X_train, X_val, y_train, y_val = train_test_split(
    combined_features_smoteenn, labels_smoteenn,
    test_size=0.2,
    random_state=42,
    stratify=labels_smoteenn
)

# Separate titles and contexts for training and validation
X_title_train, X_context_train = X_train[:, :maxlen_title], X_train[:, maxlen_title:]
X_title_val, X_context_val = X_val[:, :maxlen_title], X_val[:, maxlen_title:]

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.200d.txt', 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def word_level_attention(inputs, maxlen, name_prefix, hp):
    x = Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=True,
                  name=f'{name_prefix}_embedding')(inputs)

    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, step=1)
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=16)

    for i in range(num_lstm_layers):
        x = Bidirectional(
            LSTM(
                units=lstm_units,
                return_sequences=True,
                name=f'{name_prefix}_lstm_{i + 1}'
            )
        )(x)

        if i < num_lstm_layers - 1:
            x = Dropout(
                rate=hp.Choice(f'{name_prefix}_lstm_dropout_{i}',
                               values=[0.1, 0.2, 0.3]),
                name=f'{name_prefix}_lstm_dropout_layer_{i}'
            )(x)

    attention = Attention(name=f'{name_prefix}_attention')([x, x])
    x = GlobalAveragePooling1D(name=f'{name_prefix}_pool')(attention)
    return x


def build_model(hp):
    title_input = Input(shape=(maxlen_title,), dtype='int32', name='title_input')
    context_input = Input(shape=(maxlen_context,), dtype='int32', name='context_input')

    title_features = word_level_attention(title_input, maxlen_title, 'title', hp)
    context_features = word_level_attention(context_input, maxlen_context, 'context', hp)

    merged = tf.keras.layers.concatenate([title_features, context_features], name='merge_features')

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)

    x = merged
    for i in range(num_dense_layers):
        x = Dense(
            hp.Int(f'dense_units_{i}', min_value=32, max_value=256, step=16),
            activation='relu',
            name=f'dense_layer_{i}'
        )(x)
        x = Dropout(
            rate=hp.Choice(f'dropout_rate_{i}', values=[0.1, 0.2, 0.3, 0.4]),
            name=f'dense_dropout_layer_{i}'
        )(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[title_input, context_input], outputs=output)

    initial_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model


# Initialize tuner with val_loss as objective
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=30,
    factor=3,
    directory='han_dir',
    project_name='han_smoteen'
)

# Search for best hyperparameters
tuner.search(
    [X_title_train, X_context_train],
    y_train,
    epochs=30,
    validation_data=([X_title_val, X_context_val], y_val),
    batch_size=16,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            restore_best_weights=True
        )
    ]
)

# Get and print best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("\nBest Hyperparameters:")
print("-" * 50)
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train the best model
model = tuner.hypermodel.build(best_hp)
history = model.fit(
    [X_title_train, X_context_train],
    y_train,
    epochs=50,
    validation_data=([X_title_val, X_context_val], y_val),
    batch_size=16,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            restore_best_weights=True
        )
    ]
)

# Evaluate the model with multiple metrics
y_pred = (model.predict([X_title_val, X_context_val]) > 0.5).astype("int32")

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("\nFinal Evaluation:")
print("-" * 50)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)