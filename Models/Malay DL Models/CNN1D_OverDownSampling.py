import os
import pandas as pd
import numpy as np
import tensorflow as tf
import malaya
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
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
    X_train_embed = np.load('X_train_embeddings.npy')
    X_test_embed = np.load('X_test_embeddings.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
else:
    print("Generating new embeddings...")
    malaya_model = malaya.embedding.huggingface(
        model='mesolitica/mistral-embedding-349m-8k-contrastive',
        revision='main'
    )
    X_train_embed = np.array([malaya_model.encode([text])[0] for text in tqdm(X_train, desc="Encoding training data")])
    np.save('X_train_embeddings.npy', X_train_embed)
    X_test_embed = np.array([malaya_model.encode([text])[0] for text in tqdm(X_test, desc="Encoding test data")])
    np.save('X_test_embeddings.npy', X_test_embed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

# Apply undersampling: Maintain 1.5:1 ratio between "real" and "fake" samples
fake_indices = np.where(y_train == 0)[0]
real_indices = np.where(y_train == 1)[0]

target_real_count = int(len(fake_indices) * 1.5)
np.random.seed(42)
undersampled_real_indices = np.random.choice(real_indices, target_real_count, replace=False)
undersampled_indices = np.concatenate([fake_indices, undersampled_real_indices])
np.random.shuffle(undersampled_indices)

X_train_embed = X_train_embed[undersampled_indices]
y_train = y_train[undersampled_indices]

# Print number of records after downsampling
print("Number of records after downsampling:")
print(f"  Real: {sum(y_train == 1)}")
print(f"  Fake: {sum(y_train == 0)}")

# Apply SMOTE to oversample the minority class ('fake')
smote = SMOTE(random_state=42)
X_train_embed, y_train = smote.fit_resample(X_train_embed, y_train)

# Print number of records after oversampling
print("\nNumber of records after oversampling:")
print(f"  Real: {sum(y_train == 1)}")
print(f"  Fake: {sum(y_train == 0)}")


# Define the CNN model with Malaya embeddings
def create_model(conv_filters=32, kernel_size=3, dropout_rate=0.5, dense_units=64, learning_rate=0.001,
                 num_conv_layers=1):
    inputs = layers.Input(shape=(X_train_embed.shape[1],))
    x = layers.Reshape((X_train_embed.shape[1], 1))(inputs)
    for _ in range(num_conv_layers):
        x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Hyperparameter tuning using KerasTuner
def build_model(hp):
    conv_filters = hp.Choice('conv_filters', values=[32, 64, 128, 256])
    kernel_size = hp.Choice('kernel_size', values=[3, 5, 7, 9])
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
    dense_units = hp.Choice('dense_units', values=[64, 128, 256, 512])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    num_conv_layers = hp.Choice('num_conv_layers', values=[2, 3, 4, 5])

    return create_model(conv_filters, kernel_size, dropout_rate, dense_units, learning_rate, num_conv_layers)


# Initialize tuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=50,
    factor=3,
    directory='cnn_dir',
    project_name='cnn_1d'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True
)

tuner.search(X_train_embed, y_train,
             epochs=30,
             validation_split=0.2,
             callbacks=[early_stopping],
             batch_size=64)

best_model = tuner.get_best_models(num_models=1)[0]

# Get and print best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print("-" * 50)
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Evaluate the model
y_pred = (best_model.predict(X_test_embed, batch_size=64) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
