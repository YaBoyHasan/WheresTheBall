import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

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

# Load the new model with custom layer
model = load_model('best_model.keras', custom_objects={'ChannelAttention': ChannelAttention})

TARGET_SIZE = (224, 224)  # match trainer input size
input_image_path = 'static/current-comp-image.jpeg'

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

img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"Input image not found: {input_image_path}")

original_h, original_w = img.shape[:2]

img_resized, scale, pad_left, pad_top = resize_with_padding(img, TARGET_SIZE)
input_tensor = img_resized.astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

pred_norm = model.predict(input_tensor)[0]

# Scale prediction back to original image coords
x_scaled = (pred_norm[0] * TARGET_SIZE[0] - pad_left) / scale
y_scaled = (pred_norm[1] * TARGET_SIZE[1] - pad_top) / scale

# Predict range due to 1 pixel offset in 224 resolution
pixel_error = 1
x_range = (
    ((pred_norm[0] * TARGET_SIZE[0] - pixel_error) - pad_left) / scale,
    ((pred_norm[0] * TARGET_SIZE[0] + pixel_error) - pad_left) / scale
)
y_range = (
    ((pred_norm[1] * TARGET_SIZE[1] - pixel_error) - pad_top) / scale,
    ((pred_norm[1] * TARGET_SIZE[1] + pixel_error) - pad_top) / scale
)

output_img = img.copy()
x_int, y_int = int(round(x_scaled)), int(round(y_scaled))

# Draw '+'
cross_size = 50
color = (0, 255, 0)
thickness = 5
cv2.line(output_img, (x_int - cross_size, y_int), (x_int + cross_size, y_int), color, thickness)
cv2.line(output_img, (x_int, y_int - cross_size), (x_int, y_int + cross_size), color, thickness)

# Draw error range box
x_min, x_max = int(round(x_range[0])), int(round(x_range[1]))
y_min, y_max = int(round(y_range[0])), int(round(y_range[1]))
cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

output_path = os.path.join('static', 'scaled_prediction_new.jpeg')
cv2.imwrite(output_path, output_img)
print(f"Saved annotated prediction to {output_path}")

# Save prediction coords and range to txt
text = (
    f"Original image size: {original_w}x{original_h}\n"
    f"Predicted coords (x, y): ({x_scaled:.1f}, {y_scaled:.1f})\n"
    f"X Range: {x_min} to {x_max}\n"
    f"Y Range: {y_min} to {y_max}\n"
    f"Scale used: {scale:.6f}\n"
)
with open('static/predicted_result_new.txt', 'w') as f:
    f.write(text)
print("Prediction data saved to static/predicted_result_new.txt")
