import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras_tuner as kt
import tensorflow_text as text
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample, class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

# Parameters
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
EPOCHS = 30


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
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
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')

    # BERT layers
    preprocessing_layer = bert_preprocess(text_input)
    encoder_outputs = bert_encoder(preprocessing_layer)

    # Use both pooled output and sequence output
    pooled_output = encoder_outputs['pooled_output']
    sequence_output = encoder_outputs['sequence_output']

    # Single attention layer
    attention = layers.MultiHeadAttention(
        num_heads=hp.Choice("num_attention_heads", values=[4, 6, 8]),
        key_dim=64
    )(sequence_output, sequence_output)
    attention = layers.GlobalAveragePooling1D()(attention)

    # Combine features
    combined = layers.Concatenate()([pooled_output, attention])

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

    output = layers.Dense(1, activation='sigmoid')(combined)
    model = tf.keras.Model(inputs=[text_input], outputs=[output])

    # Define the learning rate schedule
    learning_rate = CustomSchedule(
        initial_learning_rate=hp.Choice("learning_rate", values=[1e-5, 2e-5, 3e-5]),
        warmup_steps=1000,
        decay_steps=5000
    )

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')]
    )

    return model


# Combine the 'Tokenized_Title' and 'Tokenized_Full_Context' columns
def combine_text(row):
    tokenized_title = ast.literal_eval(row['Tokenized_Title'])
    tokenized_full_context = ast.literal_eval(row['Tokenized_Full_Context'])

    # Add special tokens to differentiate title and content
    title_text = '[TITLE] ' + ' '.join(tokenized_title)
    content_text = ' [CONTENT] ' + ' '.join(tokenized_full_context)

    # Truncate content if needed while preserving complete sentences
    max_content_tokens = MAX_SEQ_LENGTH - len(tokenized_title) - 3
    content_words = content_text.split()
    if len(content_words) > max_content_tokens:
        content_text = ' '.join(content_words[:max_content_tokens])

    return title_text + content_text


# Load and preprocess data
file_path = os.path.join("..", "Dataset", "Processed_Dataset_EN.csv")
data = pd.read_csv(file_path)
data['combined_text'] = data.apply(combine_text, axis=1)
data['target'] = data['classification_result'].apply(lambda x: 1 if x == 'real' else 0)


def downsample_data(X_combined, y):
    data_combined = pd.DataFrame({'text': X_combined, 'label': y})
    data_majority = data_combined[data_combined.label == 1]
    data_minority = data_combined[data_combined.label == 0]

    # Adjust ratio to 1.5:1 instead of 1:1 to preserve more data
    n_samples = int(len(data_minority) * 1.5)
    data_majority_downsampled = resample(data_majority,
                                         replace=False,
                                         n_samples=n_samples,
                                         random_state=42)

    data_balanced = pd.concat([data_minority,
                               data_majority_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return data_balanced['text'].tolist(), data_balanced['label'].tolist()


# Perform downsampling before splitting
texts = data['combined_text'].tolist()
labels = data['target'].tolist()
texts, labels = downsample_data(texts, labels)

# Splitting the downsampled data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Debugging alignment
for text, label in zip(train_texts[:5], train_labels[:5]):
    print(f"Text: {text[:50]}... Label: {label}")

# Ensure labels are aligned with their texts
assert len(train_texts) == len(train_labels), "Mismatch in texts and labels length."
assert len(test_texts) == len(test_labels), "Mismatch in texts and labels length."

# Compute class weights dynamically
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("Computed class weights:", class_weights)

# Define preprocess and encoder models from TensorFlow Hub
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# Preprocessing layer
bert_preprocess = hub.KerasLayer(tfhub_handle_preprocess)
bert_encoder = hub.KerasLayer(tfhub_handle_encoder)

# Debugging BERT layers
sample_text = "Sample input text for BERT debugging."
preprocessed = bert_preprocess([sample_text])
# Preprocessed is a dictionary; inspect keys and shapes
print("Preprocessed shapes:")
for key, value in preprocessed.items():
    print(f"{key}: shape={value.shape}")
# Pass the preprocessed dictionary to the encoder
encoded = bert_encoder(preprocessed)
# Debugging encoded output
print("Encoded output keys:", encoded.keys())
print("Pooled output shape:", encoded["pooled_output"].shape)
print("Sequence output shape:", encoded["sequence_output"].shape)

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)) \
    .shuffle(len(train_texts), seed=42) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels)) \
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

# Search for best hyperparameters
tuner.search(train_ds,
             validation_data=test_ds,
             epochs=EPOCHS,
             callbacks=callbacks,
             class_weight=class_weights)

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
    callbacks=callbacks,
    class_weight=class_weights
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
