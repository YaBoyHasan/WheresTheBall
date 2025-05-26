import os
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

# ====== CONFIG ======
IMAGE_DIR = 'static/comp-images'
DB_PATH = 'database.db'
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
# ====================

# ChannelAttention layer (same as your improved)
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense = tf.keras.layers.Dense(channels // 8, activation='relu')
        self.dense_avg = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.dense_max = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.dense_avg(self.shared_dense(avg_pool))
        max_out = self.dense_max(self.shared_dense(max_pool))
        weights = tf.sigmoid(avg_out + max_out)
        return inputs * weights

# Dummy pose extractor (replace with your real one)
def extract_pose_keypoints(image):
    # returns fixed zero vector for now, shape (34,)
    return np.zeros(34, dtype=np.float32)

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
poses = []
coords = []

for judges_x, judges_y, img_file in data_rows:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Missing image {img_file}, skipping.")
        continue
    
    original_h, original_w = img.shape[:2]

    # Normalize coords [0,1] based on original size
    norm_x = judges_x / original_w
    norm_y = judges_y / original_h

    img_resized, scale, pad_left, pad_top = resize_with_padding(img, INPUT_SIZE)

    # Adjust normalized coords for resized+padded image
    norm_x_resized = (norm_x * original_w * scale + pad_left) / INPUT_SIZE[0]
    norm_y_resized = (norm_y * original_h * scale + pad_top) / INPUT_SIZE[1]

    # Pose extraction on padded+resized img (dummy here)
    pose_kp = extract_pose_keypoints(img_resized)

    images.append(img_resized)
    poses.append(pose_kp)
    coords.append([norm_x_resized, norm_y_resized])

images = np.array(images, dtype=np.float32) / 255.0
poses = np.array(poses, dtype=np.float32)
coords = np.array(coords, dtype=np.float32)

print(f"Loaded {len(images)} samples.")

# Train-test split
X_img_train, X_img_test, X_pose_train, X_pose_test, y_train, y_test = train_test_split(
    images, poses, coords, test_size=0.2, random_state=42
)

print("Building model...")

# MobileNetV2 backbone
base_model = MobileNetV2(input_shape=(INPUT_SIZE[1], INPUT_SIZE[0], 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs_img = Input(shape=(INPUT_SIZE[1], INPUT_SIZE[0], 3), name="image_input")
inputs_pose = Input(shape=(poses.shape[1],), name="pose_input")

x = base_model(inputs_img)
x = ChannelAttention()(x)
x = layers.GlobalAveragePooling2D()(x)

# Combine image features + pose
x = layers.Concatenate()([x, inputs_pose])

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(2, activation='sigmoid')(x)  # normalized x,y coords

model = Model(inputs=[inputs_img, inputs_pose], outputs=output)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

callbacks = [
    EarlyStopping(patience=10, monitor='val_mae', restore_best_weights=True),
    ModelCheckpoint('best_model_with_pose.keras', monitor='val_mae', save_best_only=True),
    TensorBoard(log_dir='logs_pose')
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
