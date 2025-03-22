import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, GlobalMaxPool1D, Input, Reshape
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load pre-saved embeddings and labels
X_train_embed = np.load('X_train_embeddings.npy')
X_test_embed = np.load('X_test_embeddings.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')


# Custom F1 score metric
def f1_m(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# Define the BiLSTM model with adjustments
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train_embed.shape[1],)))
    # Reshape to add the timesteps dimension
    model.add(Reshape((X_train_embed.shape[1], 1)))
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
        metrics=[f1_m])
    return model


# Split embeddings into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_embed, y_train, test_size=0.2, random_state=42,
                                                  stratify=y_train)

# Hyperparameter tuning with Hyperband and early stopping
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f1_m", direction='max'),
    max_epochs=50,  # Increase max epochs to allow more exploration
    factor=3,
    directory='bilstm_dir',
    project_name='bilstm_embeddings'
)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=128,
             callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Build and train the best model
model = tuner.hypermodel.build(best_hp)
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=128,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Evaluate the model
y_pred = (model.predict(X_test_embed) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

