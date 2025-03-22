import os
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import keras_tuner as kt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load precomputed embeddings
X_train_embed = np.load('X_train_embeddings.npy')
X_test_embed = np.load('X_test_embeddings.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Split train embeddings into train and validation sets
X_train_embed, X_val_embed, y_train, y_val = train_test_split(
    X_train_embed, y_train, test_size=0.2, random_state=42, stratify=y_train
)


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


# Define LSTM model
def build_model(hp):
    model = Sequential()
    # First LSTM layer with batch normalization and dropout
    model.add(LSTM(units=hp.Int('units', min_value=64, max_value=256, step=64),
                   input_shape=(X_train_embed.shape[1], 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Choice('dropout', values=[0.3, 0.4, 0.5])))
    # Second LSTM layer
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=128, step=32), return_sequences=True))
    model.add(GlobalMaxPool1D())
    # Fully connected layers with dropout
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=128, step=16), activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout_dense', values=[0.3, 0.4, 0.5])))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name="auc"), f1_m])
    return model


class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction='max'),
    max_epochs=50,
    factor=3,
    directory='lstm_dir',
    project_name='lstm_with_embeddings'
)

# Reshape embeddings for LSTM
X_train_embed = X_train_embed.reshape(-1, X_train_embed.shape[1], 1)
X_val_embed = X_val_embed.reshape(-1, X_train_embed.shape[1], 1)

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_f1_m', factor=0.5, patience=2, min_lr=1e-6, mode='max')

# Perform hyperparameter tuning
tuner.search(X_train_embed, y_train, epochs=50, validation_data=(X_val_embed, y_val), batch_size=64,
             callbacks=[EarlyStopping(monitor='val_f1_m', patience=5, mode='max'), reduce_lr],
             class_weight=class_weights)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train the best model
model = tuner.hypermodel.build(best_hp)
history = model.fit(X_train_embed, y_train, epochs=50, validation_data=(X_val_embed, y_val), batch_size=64,
                    callbacks=[EarlyStopping(monitor='val_f1_m', patience=5, mode='max'), reduce_lr],
                    class_weight=class_weights)

# Predict and evaluate
X_test_embed = X_test_embed.reshape(-1, X_test_embed.shape[1], 1)
y_pred = (model.predict(X_test_embed) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

