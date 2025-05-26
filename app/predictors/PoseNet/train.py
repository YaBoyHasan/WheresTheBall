import os
import random
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

# =============== CONFIG ===============
IMAGE_DIR = 'static/comp-images'
DB_PATH = 'database.db'
TARGET_SIZE = (224, 224)  # width, height for MobileNetV2 default
BATCH_SIZE = 8
EPOCHS = 10
# =====================================

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

def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT JudgesX, JudgesY, ImageFileName 
        FROM BotbComps 
        WHERE JudgesX IS NOT NULL AND JudgesY IS NOT NULL AND ImageFileName IS NOT NULL
    """)
    rows = c.fetchall()
    conn.close()
    return rows

def extract_pose_keypoints(image):
    # Dummy stub for pose keypoints extraction - replace with real OpenPose or MoveNet model
    # Return a fixed-size vector of keypoints (e.g., 17 keypoints * 2 = 34 values normalized [0,1])
    # For now, return zeros
    return np.zeros(34, dtype=np.float32)

def augment(image, coord):
    h, w = image.shape[:2]
    coord = coord.copy()
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        coord[0] = w - coord[0]
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    coord[0] = np.clip(coord[0], 0, w - 1)
    coord[1] = np.clip(coord[1], 0, h - 1)
    return image, coord

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size[1] / h, target_size[0] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h = target_size[1] - new_h
    pad_w = target_size[0] - new_w
    pad_top, pad_left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded, scale, pad_left, pad_top

print("Loading data from database...")
data_rows = load_data_from_db()

images = []
coords = []
poses = []

for judges_x, judges_y, img_file in data_rows:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Missing image {img_file}, skipping.")
        continue

    img, coord_aug = augment(img, [judges_x, judges_y])
    img_resized, scale, pad_left, pad_top = resize_with_padding(img, TARGET_SIZE)

    # Normalize coordinates for ball location
    x_norm = (coord_aug[0] * scale + pad_left) / TARGET_SIZE[0]
    y_norm = (coord_aug[1] * scale + pad_top) / TARGET_SIZE[1]

    # Extract pose keypoints (dummy here, replace with real model)
    pose_kp = extract_pose_keypoints(img_resized)

    images.append(img_resized)
    coords.append([x_norm, y_norm])
    poses.append(pose_kp)

images = np.array(images, dtype=np.float32) / 255.0
coords = np.array(coords, dtype=np.float32)
poses = np.array(poses, dtype=np.float32)

print(f"Loaded {len(images)} samples.")

# Train-test split
X_img_train, X_img_test, X_pose_train, X_pose_test, y_train, y_test = train_test_split(
    images, poses, coords, test_size=0.2, random_state=42
)

print("Building model...")

# Image branch
img_input = layers.Input(shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3))
base_model = MobileNetV2(
    input_shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
x = base_model(img_input)
x = ChannelAttention()(x)
x = layers.GlobalAveragePooling2D()(x)

# Pose branch
pose_input = layers.Input(shape=(poses.shape[1],))
p = layers.Dense(128, activation='relu')(pose_input)
p = layers.Dropout(0.3)(p)

# Combine branches
combined = layers.concatenate([x, p])
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Dropout(0.5)(combined)
outputs = layers.Dense(2, activation='sigmoid')(combined)  # ball coords normalized

model = Model([img_input, pose_input], outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(patience=10, monitor='val_mae', restore_best_weights=True),
    ModelCheckpoint('best_model_with_pose.keras', monitor='val_mae', save_best_only=True),
    TensorBoard(log_dir='logs')
]

print("Starting training...")
model.fit(
    [X_img_train, X_pose_train], y_train,
    validation_data=([X_img_test, X_pose_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("Training finished! Model saved as best_model_with_pose.keras")
