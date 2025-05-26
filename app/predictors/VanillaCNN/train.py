import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from config import Config

class ChannelAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense = layers.Dense(channels // 8, activation='relu')
        self.dense_avg = layers.Dense(channels, activation='sigmoid')
        self.dense_max = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.dense_avg(self.shared_dense(avg_pool))
        max_out = self.dense_max(self.shared_dense(max_pool))
        weights = tf.sigmoid(avg_out + max_out)
        return inputs * weights

# Load preprocessed data
data = np.load(Config.PROCESSED_DATA_PATH)  # e.g., 'botb-data/processed_data.npz'
images, coords = data['images'], data['coords']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, coords, test_size=0.2, random_state=42)

# Build model
base_model = MobileNetV2(input_shape=Config.TARGET_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=Config.TARGET_SHAPE)
x = base_model(inputs)
x = ChannelAttention()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(patience=5, monitor='val_mae', restore_best_weights=True),
    ModelCheckpoint(Config.MODEL_FILE, monitor='val_mae', save_best_only=True),
    TensorBoard(log_dir=Config.LOG_DIR)
]

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("Training complete. Saved to", Config.MODEL_FILE)