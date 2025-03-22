import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D, \
    Input
from tensorflow.keras.models import Model
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
df = pd.read_csv(file_path)

# Extract and preprocess input attributes
titles = df['Tokenized_Title'].apply(ast.literal_eval).tolist()
contexts = df['Tokenized_Full_Context'].apply(ast.literal_eval).tolist()
labels = df['classification_result'].apply(lambda x: 1 if x == 'real' else 0).values

# Convert tokenized lists to strings
title_texts = [" ".join(title) for title in titles]
context_texts = [" ".join(context) for context in contexts]

# Initialize and fit tokenizer separately for titles and contexts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(title_texts + context_texts)

# Convert texts to sequences
title_sequences = tokenizer.texts_to_sequences(title_texts)
context_sequences = tokenizer.texts_to_sequences(context_texts)

# Pad sequences with appropriate maxlen
maxlen_title = 20
maxlen_context = 400
padded_titles = pad_sequences(title_sequences, maxlen=maxlen_title)
padded_contexts = pad_sequences(context_sequences, maxlen=maxlen_context)

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
    x_pooled = GlobalAveragePooling1D(name=f'{name_prefix}_pool')(attention)
    return x_pooled, attention


def build_model(hp):
    title_input = Input(shape=(maxlen_title,), dtype='int32', name='title_input')
    context_input = Input(shape=(maxlen_context,), dtype='int32', name='context_input')

    # Get title and context features with attention
    title_features, title_attention = word_level_attention(title_input, maxlen_title, 'title', hp)
    context_features, context_attention = word_level_attention(context_input, maxlen_context, 'context', hp)

    # Combine features
    merged = tf.keras.layers.concatenate([title_features, context_features], name='merge_features')

    # Add dense layers
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

    # Classification output
    output = Dense(1, activation='sigmoid', name='output')(x)

    # Define model with attention scores as outputs
    model = Model(
        inputs=[title_input, context_input],
        outputs=[output, title_attention, context_attention]
    )

    # Compile with class weight applied only to the classification output
    initial_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss={'output': 'binary_crossentropy', 'title_attention': None, 'context_attention': None},
        loss_weights={'output': 1.0, 'title_attention': 0.0, 'context_attention': 0.0},
        metrics={'output': ['accuracy', tf.keras.metrics.AUC(name="auc"), f1_m]}
    )
    return model


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


# Split the data
X_title_train, X_title_val, X_context_train, X_context_val, y_train, y_val = train_test_split(
    padded_titles, padded_contexts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Initialize tuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction='max'),
    max_epochs=50,
    factor=3,
    directory='han_dir',
    project_name='han_ori'
)

# Search for best hyperparameters
tuner.search(
    [X_title_train, X_context_train],
    y_train,
    epochs=30,
    validation_data=([X_title_val, X_context_val], y_val),
    batch_size=32,
    callbacks=[
        EarlyStopping(
            monitor='val_f1_m',
            patience=3,
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

# Train the model with modified setup
model = tuner.hypermodel.build(best_hp)
history = model.fit(
    [X_title_train, X_context_train],
    {'output': y_train},  # Use only the main output for training
    epochs=50,
    validation_data=([X_title_val, X_context_val], {'output': y_val}),
    batch_size=32,
    callbacks=[
        EarlyStopping(
            monitor='val_output_f1_m',
            patience=5,
            mode='max',
            restore_best_weights=True
        )
    ]
)

# Save the model and tokenizer
model.save('en_han_model2.h5')
with open('en_tokenizer2.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Evaluate the model
y_pred = (model.predict([X_title_val, X_context_val], batch_size=16) > 0.5).astype("int32")
f1 = f1_score(y_val, y_pred)
print("\nFinal Evaluation:")
print("-" * 50)
print("F1 Score:", f1)