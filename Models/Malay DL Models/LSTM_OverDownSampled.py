import os
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import keras_tuner as kt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load precomputed embeddings
X_train_embed = np.load('X_train_embeddings.npy')
X_test_embed = np.load('X_test_embeddings.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

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

# Print number of records after downsampling
print("Number of records after oversampling:")
print(f"  Real: {sum(y_train == 1)}")
print(f"  Fake: {sum(y_train == 0)}")

# Split train embeddings into train and validation sets
X_train_embed, X_val_embed, y_train, y_val = train_test_split(
    X_train_embed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Reshape embeddings for LSTM
X_train_embed = X_train_embed.reshape(-1, X_train_embed.shape[1], 1)
X_val_embed = X_val_embed.reshape(-1, X_train_embed.shape[1], 1)


# Define LSTM model
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=64, max_value=256, step=64),
                   input_shape=(X_train_embed.shape[1], 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.3, 0.4, 0.5])))
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=128, step=32), return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=128, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.3, 0.4, 0.5])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Hyperparameter tuning
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=50,
    factor=3,
    directory='lstm_dir',
    project_name='lstm_with_embeddings'
)

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, mode='min')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Perform hyperparameter tuning
tuner.search(X_train_embed, y_train, epochs=50, validation_data=(X_val_embed, y_val), batch_size=64,
             callbacks=[early_stopping, reduce_lr])

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train the best model
model = tuner.hypermodel.build(best_hp)
history = model.fit(X_train_embed, y_train, epochs=50, validation_data=(X_val_embed, y_val), batch_size=64,
                    callbacks=[early_stopping, reduce_lr])

# Reshape test embeddings for LSTM
X_test_embed = X_test_embed.reshape(-1, X_test_embed.shape[1], 1)

# Predict and evaluate
y_pred = (model.predict(X_test_embed) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
