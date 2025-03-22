import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    return [text if isinstance(text, str) and text.strip() else "" for text in texts]


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
    embedding_size = None
    for i, text in enumerate(tqdm(texts, desc=description)):
        try:
            embedding = model.encode([text])[0]
            embeddings.append(embedding)
            if embedding_size is None:
                embedding_size = len(embedding)
        except Exception as e:
            print(f"Error encoding text at index {i}: {e}")
            embeddings.append(np.zeros(embedding_size or 768))
    return np.array(embeddings)


# Load or create embeddings
if os.path.exists('title_embeddings.npy') and os.path.exists('context_embeddings.npy'):
    title_embeddings = np.load('title_embeddings.npy')
    context_embeddings = np.load('context_embeddings.npy')
else:
    title_embeddings = encode_texts(title_texts, malaya_model, "Encoding titles")
    np.save('title_embeddings.npy', title_embeddings)
    context_embeddings = encode_texts(context_texts, malaya_model, "Encoding contexts")
    np.save('context_embeddings.npy', context_embeddings)

# Apply undersampling: Maintain 1.5:1 ratio between "real" and "fake" samples
fake_indices = np.where(labels == 0)[0]
real_indices = np.where(labels == 1)[0]
target_real_count = int(len(fake_indices) * 1.5)

np.random.seed(42)
undersampled_real_indices = np.random.choice(real_indices, target_real_count, replace=False)
undersampled_indices = np.concatenate([fake_indices, undersampled_real_indices])
np.random.shuffle(undersampled_indices)

title_embeddings = title_embeddings[undersampled_indices]
context_embeddings = context_embeddings[undersampled_indices]
labels = labels[undersampled_indices]

# Split data
X_title_train, X_title_val, X_context_train, X_context_val, y_train, y_val = train_test_split(
    title_embeddings, context_embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)


# Define word-level attention
def word_level_attention(inputs, name_prefix, hp):
    inputs_reshaped = tf.keras.layers.Reshape((1, inputs.shape[-1]))(inputs)
    x = inputs_reshaped
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        x = Bidirectional(
            LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)
        )(x)
        if i < hp.Int('num_lstm_layers', 1, 3) - 1:
            x = Dropout(hp.Choice('dropout_rate', [0.1, 0.2, 0.3]))(x)
    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)
    return x


# Define HAN model
def build_model(hp):
    title_input = Input(shape=(title_embeddings.shape[1],))
    context_input = Input(shape=(context_embeddings.shape[1],))
    title_features = word_level_attention(title_input, 'title', hp)
    context_features = word_level_attention(context_input, 'context', hp)
    merged = tf.keras.layers.concatenate([title_features, context_features])
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        merged = Dense(hp.Int('dense_units', 32, 256, step=32), activation='relu')(merged)
        merged = Dropout(hp.Choice('dense_dropout', [0.1, 0.2, 0.3]))(merged)
    output = Dense(1, activation='sigmoid')(merged)
    model = Model([title_input, context_input], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
    ), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=50,
    factor=3,
    directory='han_dir',
    project_name='han'
)

# Train with tuning
tuner.search([X_title_train, X_context_train], y_train,
             validation_data=([X_title_val, X_context_val], y_val),
             callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)],
             batch_size=32)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("\nBest Hyperparameters:")
print("-" * 50)
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

model = tuner.hypermodel.build(best_hp)
model.fit([X_title_train, X_context_train], y_train, validation_data=([X_title_val, X_context_val], y_val),
          epochs=50, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')])

# Evaluate the model
y_pred = (model.predict([X_title_val, X_context_val]) > 0.5).astype("int32")
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")