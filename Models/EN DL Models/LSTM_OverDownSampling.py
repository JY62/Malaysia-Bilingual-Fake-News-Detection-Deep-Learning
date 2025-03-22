import os
import ast
import pandas as pd
import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


# Perform undersampling of the real class
def downsample_data(X, y, factor):
    data_combined = pd.DataFrame({'text': X, 'label': y})
    data_majority = data_combined[data_combined.label == 1]
    data_minority = data_combined[data_combined.label == 0]

    # Undersample real class to 1.5 times the size of the fake class
    majority_target_size = int(len(data_minority) * factor)
    data_majority_downsampled = resample(data_majority,
                                         replace=False,
                                         n_samples=majority_target_size,
                                         random_state=42)

    # Combine and shuffle
    data_balanced = pd.concat([data_minority, data_majority_downsampled]).sample(frac=1, random_state=42).reset_index(
        drop=True)
    return data_balanced['text'].tolist(), data_balanced['label'].tolist()


# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
# Load the dataset
df = pd.read_csv(file_path)

# Extract input and target attributes
titles = df['Tokenized_Title'].apply(ast.literal_eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(ast.literal_eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values
texts = [" ".join(title) + " " + " ".join(context) for title, context in zip(titles, contexts)]

# Downsample real class to 1.5x fake class size
fake_class_size = len(df[df['classification_result'] == 'fake'])
texts_downsampled, labels_downsampled = downsample_data(texts, labels, factor=1.5)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_downsampled)
sequences = tokenizer.texts_to_sequences(texts_downsampled)
padded_sequences = pad_sequences(sequences, maxlen=500)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels_downsampled, test_size=0.2,
                                                  random_state=42, stratify=labels_downsampled)

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

# Convert padded sequences to GloVe embeddings
X_train_embedded = np.array([
    np.mean(
        [embedding_index.get(tokenizer.index_word.get(idx), np.zeros(embedding_dim))
         for idx in seq if idx > 0], axis=0) if any(idx > 0 for idx in seq) else np.zeros(embedding_dim)
    for seq in X_train
])

# Handle any empty embeddings (caused by missing tokens)
X_train_embedded = np.nan_to_num(X_train_embedded)

# Compute the target size based on real class downsampling
real_class_target_size = np.sum(np.array(y_train) == 1)


# Perform SMOTE on embedded text data
def apply_smote(X, y, target_size):
    # Calculate the sampling ratio for SMOTE
    sampling_ratio = target_size / np.sum(np.array(y) == 0)

    # Ensure the ratio is a valid float for SMOTE
    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# Apply SMOTE with corrected sampling ratio
X_train_smote, y_train_smote = apply_smote(X_train_embedded, np.array(y_train), target_size=real_class_target_size)


def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=500,
                        trainable=True))

    # First LSTM layer with batch normalization and dropout
    model.add(LSTM(units=hp.Int('units', min_value=64, max_value=256, step=64), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.3, 0.4, 0.5])))
    # Second LSTM layer
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=128, step=32), return_sequences=True))
    model.add(GlobalMaxPool1D())
    # Fully connected layers with dropout
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=128, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.3, 0.4, 0.5])))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Convert back to tensors for training
X_train = tf.convert_to_tensor(X_train_smote, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train_smote, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256)

# Hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction='min'),
    max_epochs=30,
    factor=3,
    directory='lstm_dir',
    project_name='lstm_downsampled'
)

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, mode='min')

# Perform hyperparameter tuning with EarlyStopping monitoring val_f1_m
tuner.search(train_dataset,
             epochs=30,
             validation_data=val_dataset,
             callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min'), reduce_lr])

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build the best model
model = tuner.hypermodel.build(best_hp)
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min'), reduce_lr])

# Predict probabilities and convert to binary predictions
y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Convert back to numpy for sklearn metrics
y_val_numpy = y_val.numpy()

# Calculate evaluation metrics
f1 = f1_score(y_val_numpy, y_pred)
accuracy = accuracy_score(y_val_numpy, y_pred)
precision = precision_score(y_val_numpy, y_pred)
recall = recall_score(y_val_numpy, y_pred)

# Print evaluation metrics
print("Model Evaluation Metrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
