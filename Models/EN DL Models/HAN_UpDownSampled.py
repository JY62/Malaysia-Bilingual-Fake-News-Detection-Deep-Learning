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
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def downsample_data(X_combined, y):
    data_combined = pd.DataFrame({'text': X_combined, 'label': y})
    data_majority = data_combined[data_combined.label == 1]
    data_minority = data_combined[data_combined.label == 0]

    # Adjust ratio to 1.5:1 to preserve more real data
    n_samples = int(len(data_minority) * 1.5)
    data_majority_downsampled = resample(data_majority,
                                         replace=False,
                                         n_samples=n_samples,
                                         random_state=42)

    data_balanced = pd.concat([data_minority,
                               data_majority_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return np.array(data_balanced['text'].tolist()), np.array(data_balanced['label'].tolist())


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

# Apply downsampling to title and context separately
titles_downsampled, labels_downsampled = downsample_data(title_texts, labels)
contexts_downsampled, _ = downsample_data(context_texts, labels)

# Initialize and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate([titles_downsampled, contexts_downsampled]))

# Convert text to sequences for numerical representation
title_sequences = tokenizer.texts_to_sequences(titles_downsampled)
context_sequences = tokenizer.texts_to_sequences(contexts_downsampled)

# Pad sequences
maxlen_title = 20
maxlen_context = 400
padded_titles = pad_sequences(title_sequences, maxlen=maxlen_title)
padded_contexts = pad_sequences(context_sequences, maxlen=maxlen_context)


def apply_smote(X, y):
    """
    Apply SMOTE to balance the dataset after downsampling.
    :param X: Input features (numerical representations like padded sequences).
    :param y: Labels.
    :return: Balanced features and labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# Apply SMOTE to numerical representations
padded_titles_smote, labels_smote_titles = apply_smote(padded_titles, labels_downsampled)
padded_contexts_smote, labels_smote_contexts = apply_smote(padded_contexts, labels_downsampled)

# Ensure labels are consistent between title and context
assert np.array_equal(labels_smote_titles, labels_smote_contexts), "Mismatch in SMOTE labels."

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
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)

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
            hp.Int(f'dense_units_{i}', min_value=32, max_value=256, step=32),
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


# Split the downsampled data
X_title_train, X_title_val, X_context_train, X_context_val, y_train, y_val = train_test_split(
    padded_titles_smote, padded_contexts_smote, labels_smote_titles,
    test_size=0.2,
    random_state=42,
    stratify=labels_smote_titles
)

# Initialize tuner with val_loss as objective
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='han_dir',
    project_name='han'
)

# Search for best hyperparameters
tuner.search(
    [X_title_train, X_context_train],
    y_train,
    epochs=50,
    validation_data=([X_title_val, X_context_val], y_val),
    batch_size=32,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
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
    batch_size=32,
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