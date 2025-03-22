import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras import layers, models
import keras_tuner as kt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
# Load the dataset
data = pd.read_csv(file_path)

# Preprocess the data
X_titles = data['Tokenized_Title'].apply(lambda x: eval(x)).tolist()
X_contexts = data['Tokenized_Full_Context'].apply(lambda x: eval(x)).tolist()
y = data['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values

# Combine titles and contexts
X_combined = [' '.join(title + context) for title, context in zip(X_titles, X_contexts)]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# Prepare tokenizer to convert textual data into numerical sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)  # Fit only on training data
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences after splitting to ensure consistent input dimensionality
max_length = 500 # max sequence length: 500 tokens
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_length)

# Create embedding matrix (capture semantic relationships between words)
embedding_dim = 200
glove_file_path = 'glove.6B.200d.txt'
embeddings_index = load_glove_embeddings(glove_file_path)
word_index = tokenizer.word_index
num_words = min(10000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# Define the CNN model with GloVe embeddings
def create_model(conv_filters=32, kernel_size=3, dropout_rate=0.5, dense_units=64, learning_rate=0.001,
                 embedding_trainable=False, num_conv_layers=1):
    # Input layer (representing padded numerical sequences of words)
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    # Embedding Layer (maps input sequence to dense vector representation)
    x = layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_length,
                         trainable=embedding_trainable)(inputs)

    # Convolutional Layer (Add specified number of Conv1D layers between 3 and 4)
    for _ in range(num_conv_layers):
        x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(x)
    # Global Max Pooling (reduce feature map dimensionality by selecting the max value from each filter's output)
    x = layers.GlobalMaxPooling1D()(x)
    # Dropout Layer (reduce overfitting)
    x = layers.Dropout(dropout_rate)(x)
    # Dense Layer (process extracted features to capture high-level representations)
    x = layers.Dense(dense_units, activation='relu')(x)
    # Output Layer (Produce probability score for binary classification)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    # Compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['val_loss'])
    return model


# Hyperparameter tuning using KerasTuner
def build_model(hp):
    conv_filters = hp.Choice('conv_filters', values=[32, 64])
    kernel_size = hp.Choice('kernel_size', values=[3, 4, 5])
    dropout_rate = hp.Choice('dropout_rate', values=[0.3, 0.4])
    dense_units = hp.Choice('dense_units', values=[64, 128])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    embedding_trainable = hp.Boolean('embedding_trainable')
    num_conv_layers = hp.Choice('num_conv_layers', values=[3, 4])

    model = create_model(conv_filters, kernel_size, dropout_rate, dense_units, learning_rate, embedding_trainable,
                         num_conv_layers)
    return model


# Initialize tuner with a fixed batch size of 64
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    directory='cnn_dir',
    project_name='cnn_1d'
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Perform hyperparameter search with a fixed batch size of 64
tuner.search(X_train_pad, y_train, epochs=30, validation_split=0.2, callbacks=[early_stopping], batch_size=64)

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print best hyperparameters
print("Best Hyperparameters:")
print(f"Convolutional Filters: {best_hyperparameters.get('conv_filters')}")
print(f"Kernel Size: {best_hyperparameters.get('kernel_size')}")
print(f"Dropout Rate: {best_hyperparameters.get('dropout_rate')}")
print(f"Dense Units: {best_hyperparameters.get('dense_units')}")
print(f"Learning Rate: {best_hyperparameters.get('learning_rate')}")
print(f"Embedding Trainable: {best_hyperparameters.get('embedding_trainable')}")
print(f"Number of Conv Layers: {best_hyperparameters.get('num_conv_layers')}")
print(f"Batch Size: 64")

# Evaluate the model on test data
y_pred = (best_model.predict(X_test_pad, batch_size=64) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')

