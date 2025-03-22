import os
import pandas as pd
import numpy as np
import tensorflow as tf
import malaya
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras import layers, models
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset Location
file_path = os.path.join("..", "Dataset", "Processed_Dataset_BM.csv")
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

# Check if the embeddings are already saved
if os.path.exists('X_train_embeddings.npy') and os.path.exists('X_test_embeddings.npy') and \
   os.path.exists('y_train.npy') and os.path.exists('y_test.npy'):
    print("Loading pre-saved embeddings and labels...")
    # Load the embeddings and labels from .npy files
    X_train_embed = np.load('X_train_embeddings.npy')
    X_test_embed = np.load('X_test_embeddings.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
else:
    print("Generating new embeddings...")
    # Load Malaya HuggingFace embedding with pinned revision
    malaya_model = malaya.embedding.huggingface(
        model='mesolitica/mistral-embedding-349m-8k-contrastive',
        revision='main'
    )

    # Encode text into embeddings with tqdm progress bar
    X_train_embed = np.array([malaya_model.encode([text])[0] for text in tqdm(X_train, desc="Encoding training data", unit="text")])
    np.save('X_train_embeddings.npy', X_train_embed)
    X_test_embed = np.array([malaya_model.encode([text])[0] for text in tqdm(X_test, desc="Encoding test data", unit="text")])
    np.save('X_test_embeddings.npy', X_test_embed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


# Define custom F1-score metric
def f1_m(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# Define the CNN model with Malaya embeddings
def create_model(conv_filters=32, kernel_size=3, dropout_rate=0.5, dense_units=64, learning_rate=0.001,
                 num_conv_layers=1):
    inputs = layers.Input(shape=(X_train_embed.shape[1],))
    x = layers.Reshape((X_train_embed.shape[1], 1))(inputs)  # Reshape for Conv1D

    # Add specified number of Conv1D layers
    for _ in range(num_conv_layers):
        x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=[f1_m])
    return model


# Hyperparameter tuning using KerasTuner
def build_model(hp):
    conv_filters = hp.Choice('conv_filters', values=[32, 64, 128, 256])
    kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 9])
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
    dense_units = hp.Choice('dense_units', values=[64, 128, 256, 512])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    num_conv_layers = hp.Choice('num_conv_layers', values=[2, 3, 4, 5])

    model = create_model(
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        learning_rate=learning_rate,
        num_conv_layers=num_conv_layers,
    )
    return model


# Initialize tuner with a fixed batch size of 64
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction="max"),
    max_epochs=50,
    factor=3,
    directory='cnn_dir',
    project_name='cnn_1d'
)

# Custom EarlyStopping with F1-score as the monitored metric
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_m',
    patience=5,
    mode='max',
    restore_best_weights=True
)

# Perform hyperparameter search with a fixed batch size of 64
tuner.search(X_train_embed, y_train,
             epochs=30,
             validation_split=0.2,
             callbacks=[early_stopping],
             batch_size=64)

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
print(f"Number of Conv Layers: {best_hyperparameters.get('num_conv_layers')}")
print(f"Batch Size: 64")

# Evaluate the model on test data
y_pred = (best_model.predict(X_test_embed, batch_size=64) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')

