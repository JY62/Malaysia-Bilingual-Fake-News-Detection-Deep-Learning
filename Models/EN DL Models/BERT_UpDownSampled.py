import os
import ast
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras_tuner as kt
import tensorflow_text as text
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from tqdm import tqdm

# Parameters
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
EPOCHS = 30


def text_to_embeddings(texts):
    """Convert text data into embeddings using the BERT encoder."""
    preprocessed_texts = bert_preprocess(tf.constant(texts))
    embeddings = bert_encoder(preprocessed_texts)["pooled_output"]
    return embeddings.numpy()


def balance_data_with_smote_and_undersampling(X, y):
    # Convert labels to numpy array
    y = np.array(y)

    # Separate the majority and minority classes
    majority_class = [text for text, label in zip(X, y) if label == 1]
    minority_class = [text for text, label in zip(X, y) if label == 0]

    # Step 1: Undersample majority class
    print("Performing undersampling...")
    n_samples_majority = int(len(minority_class) * 1.5)
    majority_class_downsampled = resample(majority_class,
                                          replace=False,
                                          n_samples=n_samples_majority,
                                          random_state=42)

    # Combine the downsampled majority class with the minority class
    X_combined = majority_class_downsampled + minority_class
    y_combined = [1] * len(majority_class_downsampled) + [0] * len(minority_class)

    # Step 2: Convert text to embeddings with progress monitoring
    print("Converting text data to embeddings...")
    start_time = time.time()
    X_embeddings = []

    with tqdm(total=len(X_combined), desc="Embedding Progress", ncols=100) as pbar:
        for i, text in enumerate(X_combined):
            embedding = text_to_embeddings([text])  # Process one text at a time
            X_embeddings.append(embedding[0])  # Append the resulting embedding

            # Update the progress bar
            pbar.update(1)

            # Estimate remaining time
            elapsed_time = time.time() - start_time
            estimated_time = (elapsed_time / (i + 1)) * (len(X_combined) - (i + 1))
            pbar.set_postfix(Estimated_time=f"{estimated_time:.2f}s")

    X_embeddings = np.array(X_embeddings)  # Convert list to numpy array

    # Step 3: Apply SMOTE to oversample the minority class
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_embeddings, y_combined)

    return X_balanced, y_balanced


class CustomSchedule(tf.keras.optimizers.scShedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup
        warmup_pct = tf.minimum(1.0, step / self.warmup_steps)
        warmup_lr = self.initial_learning_rate * warmup_pct
        # Decay
        decay_pct = tf.minimum(1.0, (step - self.warmup_steps) / self.decay_steps)
        decay_lr = self.initial_learning_rate * (1 - decay_pct)
        # Combine warmup and decay
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        return tf.maximum(lr, self.initial_learning_rate * 0.1)  # Minimum learning rate

    def get_config(self):
        return {
            "initial_learning_rate": float(self.initial_learning_rate.numpy()),
            "warmup_steps": int(self.warmup_steps.numpy()),
            "decay_steps": int(self.decay_steps.numpy())
        }


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Ensure y_true is float32
        y_true = tf.cast(y_true, tf.float32)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma) * y_true + tf.pow(y_pred, self.gamma) * (1 - y_true)

        return self.alpha * focal_weight * cross_entropy

    def get_config(self):
        return {
            "gamma": float(self.gamma),
            "alpha": float(self.alpha)
        }


def model_builder(hp):
    # Input layer now expects pre-embedded vectors (e.g., shape (None, 768))
    embedding_dim = 768
    text_input = layers.Input(shape=(embedding_dim,), dtype=tf.float32, name='embedding_input')

    # Reshape the input to be (batch_size, 1, embedding_dim) for the attention layer
    reshaped_input = layers.Reshape((1, embedding_dim))(text_input)

    # Single attention layer
    attention = layers.MultiHeadAttention(
        num_heads=hp.Choice("num_attention_heads", values=[4, 6, 8]),
        key_dim=64
    )(reshaped_input, reshaped_input)  # Apply attention to the reshaped input
    attention = layers.GlobalAveragePooling1D()(attention)

    # Combine features (e.g., from the attention output)
    combined = layers.Concatenate()([attention])

    # Single projection layer, reducing dimensionality
    combined = layers.Dense(
        hp.Choice("projection_dim", values=[128, 256]),
        activation='relu'
    )(combined)

    # Dense layers with regularization
    for i in range(hp.Int("num_dense_layers", 1, 2)):
        combined = layers.Dense(
            hp.Choice(f"dense_{i}_units", values=[128, 256]),
            activation='relu'
        )(combined)
        combined = layers.Dropout(hp.Float(f"dropout_{i}", 0.1, 0.3, step=0.1))(combined)
    # Output layer for binary classification
    output = layers.Dense(1, activation='sigmoid')(combined)
    model = tf.keras.Model(inputs=[text_input], outputs=[output])

    # Define the learning rate schedule
    learning_rate = CustomSchedule(
        initial_learning_rate=hp.Choice("learning_rate", values=[1e-5, 2e-5, 3e-5]),
        warmup_steps=1000,
        decay_steps=5000
    )

    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')]
    )

    return model


# Combine the 'Tokenized_Title' and 'Tokenized_Full_Context' columns
def combine_text(row):
    # Convert string representations of lists into actual lists
    tokenized_title = eval(row['Tokenized_Title'])
    tokenized_full_context = eval(row['Tokenized_Full_Context'])

    # Combine the title and context with special tokens
    title_text = '[TITLE] ' + ' '.join(tokenized_title)
    content_text = ' [CONTENT] ' + ' '.join(tokenized_full_context)

    # Truncate content if it exceeds the maximum sequence length
    max_content_tokens = MAX_SEQ_LENGTH - len(tokenized_title) - 3
    content_words = content_text.split()
    if len(content_words) > max_content_tokens:
        content_text = ' '.join(content_words[:max_content_tokens])

    return title_text + content_text


# Load and preprocess data
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
data = pd.read_csv(file_path)
data['combined_text'] = data.apply(combine_text, axis=1)
# Convert classification labels to binary format
data['target'] = data['classification_result'].apply(lambda x: 1 if x == 'real' else 0)

# Define preprocess and encoder models from TensorFlow Hub
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# Preprocessing layer
bert_preprocess = hub.KerasLayer(tfhub_handle_preprocess)
bert_encoder = hub.KerasLayer(tfhub_handle_encoder)

# Perform downsampling before splitting
texts = data['combined_text'].tolist()
labels = data['target'].tolist()
X_balanced, y_balanced = balance_data_with_smote_and_undersampling(texts, labels)

# Splitting the downsampled data
train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_embeddings, train_labels)) \
    .shuffle(len(train_embeddings), seed=42) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_embeddings, test_labels)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'bm_best_model.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
]

# Initialize the tuner with Hyperband
tuner = kt.Hyperband(
    model_builder,
    objective=kt.Objective("val_loss", direction='min'),
    max_epochs=EPOCHS,
    factor=3,
    directory="bert_dir",
    project_name="bert"
)

print("Start Model Training")

# Search for best hyperparameters
tuner.search(train_ds,
             validation_data=test_ds,
             epochs=EPOCHS,
             callbacks=callbacks)

# Get the best model hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Display the best hyperparameters
print("Best hyperparameters:", best_hps.values)

# Train the model with best hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=callbacks
)

# Evaluate model on test set
# Use different thresholds to find optimal classification threshold
thresholds = [0.3, 0.4, 0.5]
for threshold in thresholds:
    predictions = model.predict(test_ds)
    predicted_classes = np.where(predictions > threshold, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes)
    recall = recall_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes)

    print(f"\nMetrics at threshold {threshold}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(classification_report(test_labels, predicted_classes, target_names=['fake', 'real'], zero_division=0))