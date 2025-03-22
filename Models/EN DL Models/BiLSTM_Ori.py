import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import keras_tuner as kt
import tensorflow as tf
import ast

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

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.200d.txt', 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=400)

# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def build_model(hp):
    model = Sequential()
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=400,
                        trainable=True))

    # Bidirectional LSTM Layer (both forward and backward)
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True)))
    # Global Max Pooling (reduce feature map dimensionality by selecting the max value from each filter's output)
    model.add(GlobalMaxPool1D())
    # Dropout (Regularize model by randomly deactivating neurons)
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.2, 0.3, 0.4])))
    # Dense Layer (Introduce non-linearity, support capturing complex relationships within the input data)
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.2, 0.3, 0.4])))
    # Output layer (Generate probability score)
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name="auc"), f1_m])
    return model


# Custom F1 score metric for validation monitoring
def f1_m(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction='max'),
    max_epochs=30,
    factor=3,
    directory='bilstm_dir',
    project_name='bilstm'
)

X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42,
                                                  stratify=labels)

# Perform hyperparameter tuning with EarlyStopping monitoring val_f1_m
tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=128,
             callbacks=[EarlyStopping(monitor='val_f1_m', patience=3, mode='max')],
             class_weight=class_weights)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build the best model
model = tuner.hypermodel.build(best_hp)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=128,
                    callbacks=[EarlyStopping(monitor='val_f1_m', patience=3, mode='max')],
                    class_weight=class_weights)

# Predict and evaluate
y_pred = (model.predict(X_val) > 0.5).astype("int32")
f1 = f1_score(y_val, y_pred)
print("F1 Score:", f1)

