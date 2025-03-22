import os
import pandas as pd
import numpy as np
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
import ast
import tensorflow as tf
import keras_tuner as kt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
# Load the dataset
df = pd.read_csv(file_path)

# Extract input and target attributes
titles = df['Tokenized_Title'].apply(ast.literal_eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(ast.literal_eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values
texts = [" ".join(title) + " " + " ".join(context) for title, context in zip(titles, contexts)]


# Downsampling function
def downsample_data(X, y, majority_class=1, minority_class=0):
    data = pd.DataFrame({'text': X, 'label': y})
    data_majority = data[data.label == majority_class]
    data_minority = data[data.label == minority_class]

    # Downsample the majority class to 1.5x the minority class size
    downsampled_majority = resample(
        data_majority,
        replace=False,
        n_samples=int(len(data_minority) * 1.5),
        random_state=42
    )

    # Combine and shuffle
    data_balanced = pd.concat([data_minority, downsampled_majority]).sample(frac=1, random_state=42).reset_index(drop=True)
    return data_balanced['text'].tolist(), data_balanced['label'].tolist()


# Step 1: Downsampling
texts_downsampled, labels_downsampled = downsample_data(texts, labels)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_downsampled)
sequences = tokenizer.texts_to_sequences(texts_downsampled)
padded_sequences = pad_sequences(sequences, maxlen=500)

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

# Step 2: Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(padded_sequences, labels_downsampled)

# Step 3: Split data
X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)


# Build the model using Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=500,
                        trainable=True))

    # First LSTM layer
    model.add(LSTM(units=hp.Int('units', min_value=64, max_value=256, step=64), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.3, 0.4, 0.5])))

    # Second LSTM layer
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=128, step=32), return_sequences=True))
    model.add(GlobalMaxPool1D())

    # Fully connected layers
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=128, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.3, 0.4, 0.5])))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction='min'),
    max_epochs=30,
    factor=3,
    directory='lstm_dir',
    project_name='lstm_smote'
)

# TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, mode='min')

# Tuner search
tuner.search(train_dataset, validation_data=val_dataset, epochs=30, callbacks=[early_stopping, reduce_lr])

# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train final model
model = tuner.hypermodel.build(best_hp)
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=50,
                    callbacks=[early_stopping, reduce_lr])

# Model evaluation
y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Metrics
f1 = f1_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("Model Evaluation Metrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
