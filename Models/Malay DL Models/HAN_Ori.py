import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import malaya
from tqdm import tqdm
import keras_tuner as kt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_BM.csv")
df = pd.read_csv(file_path)

# Preprocess data
titles = df['Tokenized_Title'].apply(eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values

# Combine titles and contexts for Malaya embedding
title_texts = [" ".join(title) for title in titles]
context_texts = [" ".join(context) for context in contexts]


# Clean and ensure all inputs are strings
def clean_texts(texts):
    cleaned = []
    for text in texts:
        if isinstance(text, str) and text.strip():  # Ensure it's a non-empty string
            cleaned.append(text)
        else:
            cleaned.append("")  # Replace invalid entries with empty strings
    return cleaned


title_texts = clean_texts(title_texts)
context_texts = clean_texts(context_texts)

# Load Malaya HuggingFace embedding
malaya_model = malaya.embedding.huggingface(
    model='mesolitica/mistral-embedding-349m-8k-contrastive',
    revision='main'
)


# Generate embeddings with error handling
def encode_texts(texts, model, description):
    embeddings = []
    embedding_size = None  # Initialize embedding size as None
    for i, text in enumerate(tqdm(texts, desc=description)):
        try:
            embedding = model.encode([text])[0]
            embeddings.append(embedding)

            # Dynamically set the embedding size if not already set
            if embedding_size is None:
                embedding_size = len(embedding)
        except Exception as e:
            print(f"Error encoding text at index {i}: {e}")
            print(f"Problematic text: {title_texts[i]}")
            print(f"Corresponding row in the dataset: {df.iloc[i]}")
            # Use the determined embedding size or default to 768 (commonly used size)
            embeddings.append(np.zeros(embedding_size or 768))
    return np.array(embeddings)


if os.path.exists('title_embeddings.npy') and os.path.exists('context_embeddings.npy'):
    print("Loading pre-saved embeddings...")
    title_embeddings = np.load('title_embeddings.npy')
    context_embeddings = np.load('context_embeddings.npy')
else:
    print("Generating new embeddings...")
    title_embeddings = encode_texts(title_texts, malaya_model, "Encoding titles")
    np.save('title_embeddings.npy', title_embeddings)

    context_embeddings = encode_texts(context_texts, malaya_model, "Encoding contexts")
    np.save('context_embeddings.npy', context_embeddings)

# Split data
X_title_train, X_title_val, X_context_train, X_context_val, y_train, y_val = train_test_split(
    title_embeddings, context_embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}


# Custom F1 score metric
def f1_m(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# Define word-level attention with Malaya embeddings
def word_level_attention(inputs, name_prefix, hp):
    # Expand dims to add a timestep dimension
    inputs_reshaped = tf.keras.layers.Reshape((1, inputs.shape[-1]))(inputs)

    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, step=1)
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)

    x = inputs_reshaped
    for i in range(num_lstm_layers):
        x = Bidirectional(
            LSTM(units=lstm_units, return_sequences=True, name=f'{name_prefix}_lstm_{i + 1}')
        )(x)
        if i < num_lstm_layers - 1:
            x = Dropout(
                rate=hp.Choice(f'{name_prefix}_lstm_dropout_{i}', values=[0.1, 0.2, 0.3]),
                name=f'{name_prefix}_lstm_dropout_layer_{i}'
            )(x)

    attention = Attention(name=f'{name_prefix}_attention')([x, x])
    x = GlobalAveragePooling1D(name=f'{name_prefix}_pool')(attention)
    return x


# Define the model
def build_model(hp):
    title_input = Input(shape=(title_embeddings.shape[1],), name='title_input')
    context_input = Input(shape=(context_embeddings.shape[1],), name='context_input')

    # Call the attention layers
    title_features = word_level_attention(title_input, 'title', hp)
    context_features = word_level_attention(context_input, 'context', hp)

    # Merge features and proceed with dense layers
    merged = tf.keras.layers.concatenate([title_features, context_features], name='merge_features')

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
    x = merged
    for i in range(num_dense_layers):
        x = Dense(
            hp.Int(f'dense_units_{i}', min_value=32, max_value=256, step=32), # Reduce step to 16
            activation='relu',
            name=f'dense_layer_{i}'
        )(x)
        x = Dropout(
            rate=hp.Choice(f'dropout_rate_{i}', values=[0.1, 0.2, 0.3, 0.4]),
            name=f'dense_dropout_layer_{i}'
        )(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[title_input, context_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name="auc"), f1_m]
    )
    return model


# Hyperparameter tuning with KerasTuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction='max'),
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
    batch_size=32, # change to 16
    callbacks=[
        EarlyStopping(
            monitor='val_f1_m',
            patience=5,
            mode='max',
            restore_best_weights=True
        )
    ],
    class_weight=class_weights
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
            monitor='val_f1_m',
            patience=5,
            mode='max',
            restore_best_weights=True
        )
    ],
    class_weight=class_weights
)

# Evaluate the model
y_pred = (model.predict([X_title_val, X_context_val]) > 0.5).astype("int32")
f1 = f1_score(y_val, y_pred)
print("\nFinal Evaluation:")
print("-" * 50)
print("F1 Score:", f1)


