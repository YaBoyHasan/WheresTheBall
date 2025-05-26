import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# SpatialAttention layer like in trainer.py
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

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

# Path configs
MODEL_PATH = 'best_model_spatial_attention.keras'
INPUT_IMAGE_PATH = 'static/dc2025.jpg'
OUTPUT_IMAGE_PATH = 'static/scaled_prediction_spatial_attention.jpeg'
OUTPUT_TEXT_PATH = 'static/predicted_result_spatial_attention.txt'
INPUT_SIZE = (224, 224)

# Load model with custom layer
model = load_model(MODEL_PATH, custom_objects={'SpatialAttention': SpatialAttention})

img = cv2.imread(INPUT_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")

original_h, original_w = img.shape[:2]

img_resized, scale, pad_left, pad_top = resize_with_padding(img, INPUT_SIZE)
input_img = img_resized.astype(np.float32) / 255.0
input_img = np.expand_dims(input_img, axis=0)

# Dummy pose input same shape as training (34 zeros)
pose_input = np.zeros((1, 34), dtype=np.float32)

# Predict normalized coords [0,1]
pred_norm = model.predict([input_img, pose_input])[0]

# Scale prediction back to original image coords
x_scaled = (pred_norm[0] * INPUT_SIZE[0] - pad_left) / scale
y_scaled = (pred_norm[1] * INPUT_SIZE[1] - pad_top) / scale

# Prediction range for ~1 pixel margin in resized 224x224 space
pixel_error = 1
x_range = (
    ((pred_norm[0] * INPUT_SIZE[0] - pixel_error) - pad_left) / scale,
    ((pred_norm[0] * INPUT_SIZE[0] + pixel_error) - pad_left) / scale
)
y_range = (
    ((pred_norm[1] * INPUT_SIZE[1] - pixel_error) - pad_top) / scale,
    ((pred_norm[1] * INPUT_SIZE[1] + pixel_error) - pad_top) / scale
)

output_img = img.copy()
x_int, y_int = int(round(x_scaled)), int(round(y_scaled))

cross_size = 50
color = (0, 255, 0)
thickness = 5
cv2.line(output_img, (x_int - cross_size, y_int), (x_int + cross_size, y_int), color, thickness)
cv2.line(output_img, (x_int, y_int - cross_size), (x_int, y_int + cross_size), color, thickness)

x_min, x_max = int(round(x_range[0])), int(round(x_range[1]))
y_min, y_max = int(round(y_range[0])), int(round(y_range[1]))
cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

cv2.imwrite(OUTPUT_IMAGE_PATH, output_img)
print(f"Saved annotated prediction to {OUTPUT_IMAGE_PATH}")

text = (
    f"Original image size: {original_w}x{original_h}\n"
    f"Predicted coords (x, y): ({x_scaled:.1f}, {y_scaled:.1f})\n"
    f"X Range: {x_min} to {x_max}\n"
    f"Y Range: {y_min} to {y_max}\n"
    f"Scale used: {scale:.6f}\n"
)
with open(OUTPUT_TEXT_PATH, 'w') as f:
    f.write(text)
print(f"Prediction data saved to {OUTPUT_TEXT_PATH}")
