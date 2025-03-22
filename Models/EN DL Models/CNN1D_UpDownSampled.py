import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras import layers, models, regularizers
from sklearn.utils import resample
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def downsample_data(X_combined, y):
    data_combined = pd.DataFrame({'text': X_combined, 'label': y})
    data_majority = data_combined[data_combined.label == 1]
    data_minority = data_combined[data_combined.label == 0]
    data_majority_downsampled = resample(data_majority,
                                         replace=False,
                                         n_samples=int(len(data_minority)*1.5),
                                         random_state=42)
    data_balanced = pd.concat([data_minority,
                               data_majority_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return data_balanced['text'].tolist(), data_balanced['label'].tolist()


# Define a function for embedding text data and applying SMOTE
def embed_and_oversample(X_combined, y):
    # Convert text data into numerical vectors using TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to prevent memory issues
    X_tfidf = vectorizer.fit_transform(X_combined)

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

    return X_resampled, y_resampled, vectorizer


# Apply downsampling to majority class
X_combined_balanced, y_balanced = downsample_data(X_combined, y)

# Split the dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(X_combined_balanced, y_balanced, test_size=0.2, random_state=42)

# Print the number of records of each class
unique, counts = np.unique(y_balanced, return_counts=True)
print("Number of records after downsampling:")
for label, count in zip(unique, counts):
    label_name = 'Real' if label == 1 else 'Fake'
    print(f"{label_name}: {count}")


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


# Tokenize and pad sequences for CNN
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train_text)  # Use original text data for tokenizer
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Padding sequences
max_length = 1000
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_length)

# Create embedding matrix
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
                 embedding_trainable=True, num_conv_layers=1):
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    x = layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_length,
                         trainable=embedding_trainable)(inputs)

    # Apply spatial dropout to embedding layer
    x = layers.SpatialDropout1D(rate=dropout_rate)(x)

    # Use multiple kernel sizes to capture features at different scales
    conv_outputs = []
    for k in [3, 5, 7]:
        conv = layers.Conv1D(filters=conv_filters, kernel_size=k, activation='relu',
                             kernel_regularizer=regularizers.l2(0.005))(x)
        conv_outputs.append(layers.GlobalMaxPooling1D()(conv))

    # Concatenate convolution outputs
    x = layers.concatenate(conv_outputs)

    # Dense layer with dropout
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
    x = layers.Dropout(rate=dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.9
    )

    # Compile the model
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Hyperparameter tuning using KerasTuner
def build_model(hp):
    conv_filters = hp.Int('conv_filters', min_value=16, max_value=128, step=16)
    kernel_size = hp.Int('kernel_size', min_value=2, max_value=7, step=1)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
    dense_units = hp.Int('dense_units', min_value=32, max_value=256, step=32)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    embedding_trainable = hp.Boolean('embedding_trainable')
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=6, step=1)

    # Return the model with hyperparameters
    return create_model(conv_filters, kernel_size, dropout_rate, dense_units, learning_rate,
                        embedding_trainable, num_conv_layers)


# Initialize tuner with a fixed batch size of 64
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_loss', direction='min'),
    max_epochs=30,
    factor=3,
    directory='cnn_dir',
    project_name='cnn_1d'
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

# Convert the data to NumPy arrays
X_train_pad = np.array(X_train_pad)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_pad, y_train, test_size=0.2, random_state=42
)

# Perform hyperparameter search with a fixed batch size of 64
tuner.search(
    X_train_split,
    y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=30,
    callbacks=[early_stopping],
    batch_size=64
)

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

# Train the best model with class weights
best_model.fit(X_train_pad, y_train,
               epochs=50,
               batch_size=64,
               validation_data=(X_test_pad, y_test),
               callbacks=[early_stopping])

# Evaluate the model on test data
y_pred = (best_model.predict(X_test_pad, batch_size=64) > 0.5).astype("int32")

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
