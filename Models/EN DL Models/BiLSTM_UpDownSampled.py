import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D,
    Bidirectional, LayerNormalization, SpatialDropout1D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import keras_tuner as kt
import ast
import tensorflow as tf

# Enable GPU memory growth to prevent OOM errors
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
df = pd.read_csv(file_path)

# Extract and preprocess data with separate title and context embeddings
titles = df['Tokenized_Title'].apply(ast.literal_eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(ast.literal_eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values

# Combine with special separator token
texts = [" ".join(title) + " [SEP] " + " ".join(context) for title, context in zip(titles, contexts)]

# Enhanced tokenization with special tokens
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = min(max(len(seq) for seq in sequences), 512)  # Cap at 512 tokens
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Load and process GloVe embeddings with special handling
embedding_index = {}
with open('glove.6B.200d.txt', 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Prepare embedding matrix with xavier initialization for unknown words
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 200
embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))  # Xavier init
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def balance_data_with_smote(X, y):
    """
    Applies undersampling on the majority class, followed by SMOTE on the minority class.
    """
    # Combine data for undersampling
    data_combined = pd.DataFrame({'text': list(X), 'label': y})
    majority_class = data_combined[data_combined.label == 1]
    minority_class = data_combined[data_combined.label == 0]

    # Undersample the majority class to slightly favor real class
    majority_downsampled = resample(
        majority_class,
        replace=False,
        n_samples=int(len(minority_class) * 1.5),
        random_state=42
    )

    # Combine minority and undersampled majority
    balanced_data = pd.concat([minority_class, majority_downsampled]).sample(frac=1, random_state=42)
    X_balanced = np.array(balanced_data['text'].tolist())
    y_balanced = np.array(balanced_data['label'].tolist())

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_balanced, y_balanced)

    return X_smote, y_smote


# Apply balanced downsampling
padded_sequences_smote, labels_smote = balance_data_with_smote(padded_sequences, labels)


def build_model(hp):
    model = Sequential()

    # Embedding layer with spatial dropout
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
    model.add(SpatialDropout1D(hp.Choice('spatial_dropout', values=[0.1, 0.2, 0.3])))

    # Modified LSTM layers for cuDNN compatibility
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)

    # First Bi-LSTM layer
    model.add(Bidirectional(LSTM(
        units=lstm_units,
        return_sequences=True,
        # Remove recurrent_dropout for cuDNN compatibility
        # Remove dropout for cuDNN compatibility
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal'
    )))
    model.add(LayerNormalization())
    model.add(Dropout(hp.Choice('dropout_1', values=[0.1, 0.2, 0.3])))

    # Optional second Bi-LSTM layer based on hyperparameter
    if hp.Choice('use_second_lstm', values=[0, 1]):
        model.add(Bidirectional(LSTM(
            units=lstm_units // 2,  # Reduce units for second layer
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )))
        model.add(LayerNormalization())
        model.add(Dropout(hp.Choice('dropout_2', values=[0.1, 0.2, 0.3])))

    # Global pooling
    model.add(GlobalMaxPool1D())

    # Dense layers with batch normalization
    dense_units = hp.Int('dense_units', min_value=32, max_value=256, step=32)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(hp.Choice('dropout_dense', values=[0.2, 0.3, 0.4])))
    model.add(LayerNormalization())

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile with optimizer
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4]))

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Initialize tuner with increased max_epochs
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=30,
    factor=3,
    directory='bilstm_dir',
    project_name='bilstm_improved_cudnn'
)

# Split data with stratification
X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences_smote,
    labels_smote,
    test_size=0.2,
    random_state=42,
    stratify=labels_smote
)


# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Perform hyperparameter search
tuner.search(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    batch_size=64,
    callbacks=callbacks
)

# Get best hyperparameters and build final model
best_hp = tuner.get_best_hyperparameters()[0]
print("\nBest Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

model = tuner.hypermodel.build(best_hp)

# Train final model with cross-validation
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
y_pred = (model.predict(X_val) > 0.5).astype("int32")
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("\nEvaluation Metrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")