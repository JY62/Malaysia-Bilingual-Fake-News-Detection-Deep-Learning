from imblearn.over_sampling import SMOTE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, GlobalMaxPool1D, Input, Reshape
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import tensorflow as tf

# Load pre-saved embeddings and labels
X_train_embed = np.load('X_train_embeddings.npy')
X_test_embed = np.load('X_test_embeddings.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Apply undersampling: Maintain 1.5:1 ratio between "real" and "fake" samples
fake_indices = np.where(y_train == 0)[0]
real_indices = np.where(y_train == 1)[0]

# Calculate target size for "real" samples (1.5x the number of "fake" samples)
target_real_count = int(len(fake_indices) * 1.5)

# Randomly sample "real" indices to meet the target count
np.random.seed(42)
undersampled_real_indices = np.random.choice(real_indices, target_real_count, replace=False)

# Combine the undersampled "real" and all "fake" indices
undersampled_indices = np.concatenate([fake_indices, undersampled_real_indices])

# Shuffle the indices
np.random.shuffle(undersampled_indices)

# Create undersampled datasets
X_train = X_train_embed[undersampled_indices]
y_train = y_train[undersampled_indices]

# Apply SMOTE to oversample the minority class ('fake') to match the number of 'real'
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Split embeddings into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# Define the BiLSTM model
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    # Reshape to add the timesteps dimension
    model.add(Reshape((X_train.shape[1], 1)))
    # Add Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=64, max_value=512, step=64),
                                 return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01))))
    model.add(GlobalMaxPool1D())
    # Add Dropout and Dense layers
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.2, 0.3, 0.4])))
    model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.3, 0.4, 0.5])))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 5e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


# Hyperparameter tuning with Hyperband
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_loss", direction='min'),
    max_epochs=50,
    factor=3,
    directory='bilstm_dir',
    project_name='bilstm_embeddings'
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=128,
             callbacks=[early_stopping])

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train the best model
model = tuner.hypermodel.build(best_hp)
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=128,
          callbacks=[early_stopping])

# Reshape the test set for prediction
X_test = X_test_embed

# Predict and evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
