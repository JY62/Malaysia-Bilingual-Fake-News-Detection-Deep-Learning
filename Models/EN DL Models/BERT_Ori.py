import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Parameters
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
EPOCHS = 20


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
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma) * y_true + tf.pow(y_pred, self.gamma) * (1 - y_true)

        return self.alpha * focal_weight * cross_entropy

    def get_config(self):
        return {
            "gamma": float(self.gamma),
            "alpha": float(self.alpha)
        }


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_m', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize true positives, false positives, and false negatives
        self.true_positives = self.add_weight(
            'tp', initializer='zeros')
        self.false_positives = self.add_weight(
            'fp', initializer='zeros')
        self.false_negatives = self.add_weight(
            'fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)

        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        # Update running totals
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_negatives
        )

        # Calculate F1 using tf.math.divide_no_nan to handle edge cases
        f1 = tf.math.divide_no_nan(
            2 * precision * recall,
            precision + recall
        )
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model_builder(hp):
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')

    # BERT layers (Preprocessing Layer and Encoder Layer)
    # Preprocessing Layer: Tokenization, Lowercasing, Normalization
    preprocessing_layer = bert_preprocess(text_input)
    # BERT Encoder Layer: Generate contextual word embeddings (in both syntactic and semantic)
    encoder_outputs = bert_encoder(preprocessing_layer)

    # Use both pooled output and sequence output
    pooled_output = encoder_outputs['pooled_output'] # Fixed-length of vector representation
    sequence_output = encoder_outputs['sequence_output'] # Sequence of token embeddings

    # Single attention layer (Focus on keywords or phrases)
    attention = layers.MultiHeadAttention(
        num_heads=hp.Choice("num_attention_heads", values=[4, 6, 8]),
        key_dim=64
    )(sequence_output, sequence_output)
    # Global Average Pooling (Reduces the dimensionality of sequence output and create fixed-length representation
    # of the input text)
    attention = layers.GlobalAveragePooling1D()(attention)

    # Concatenation Layer (Combine pooled output (global information) and attention (localized focus))
    combined = layers.Concatenate()([pooled_output, attention])
    # Single projection layer, reducing dimensionality
    combined = layers.Dense(
        hp.Choice("projection_dim", values=[128, 256]),
        activation='relu'
    )(combined)
    # Dense layers with regularization (prevent overfitting)
    for i in range(hp.Int("num_dense_layers", 1, 2)):
        combined = layers.Dense(
            hp.Choice(f"dense_{i}_units", values=[128, 256]),
            activation='relu'
        )(combined)
        combined = layers.Dropout(hp.Float(f"dropout_{i}", 0.1, 0.3, step=0.1))(combined)
    # Output Layer
    output = layers.Dense(1, activation='sigmoid')(combined)
    model = tf.keras.Model(inputs=[text_input], outputs=[output])

    # Learning rate schedule
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 2e-5, 3e-5])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(curve='PR'), F1Score(threshold=0.5)]
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

# Split dataset
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['target']
)

# Define preprocess and encoder models from TensorFlow Hub
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# Preprocessing layer
bert_preprocess = hub.KerasLayer(tfhub_handle_preprocess)
bert_encoder = hub.KerasLayer(tfhub_handle_encoder)

# Prepare data
train_texts = train_data['combined_text'].tolist()
test_texts = test_data['combined_text'].tolist()
train_labels = train_data['target'].values
test_labels = test_data['target'].values

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))\
    .shuffle(10000)\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_f1_m',
        patience=3,
        restore_best_weights=True,
        mode='max'
    )
]

# Initialize the tuner with Hyperband (adaptive, memory-efficient search)
tuner = kt.Hyperband(
    model_builder,
    objective=kt.Objective("val_f1_m", direction='max'),
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
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluate model on test set using F1-score
predictions = model.predict(test_ds)
predicted_classes = np.where(predictions > 0.3, 1, 0)

# Calculate and print F1 score
f1 = f1_score(test_labels, predicted_classes)
print(f"F1 Score: {f1}")
print(classification_report(test_labels, predicted_classes, target_names=['fake', 'real'], zero_division=0))


