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

# Dummy pose extractor (replace with your real one if you want)
def extract_pose_keypoints(image):
    return np.zeros(34, dtype=np.float32)

TARGET_SIZE = (224, 224)
MODEL_PATH = 'best_model_with_pose.keras'
INPUT_IMAGE_PATH = 'static/current-comp-image.jpeg'

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

def predict_ball_location(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")

    original_h, original_w = img.shape[:2]
    img_padded, scale, pad_left, pad_top = resize_with_padding(img, TARGET_SIZE)
    img_norm = img_padded.astype(np.float32) / 255.0

    pose_kp = extract_pose_keypoints(img_padded)

    img_input = np.expand_dims(img_norm, axis=0)
    pose_input = np.expand_dims(pose_kp, axis=0)

    pred_norm = model.predict([img_input, pose_input])[0]

    # Convert normalized prediction back to original image coords
    x_scaled = (pred_norm[0] * TARGET_SIZE[0] - pad_left) / scale
    y_scaled = (pred_norm[1] * TARGET_SIZE[1] - pad_top) / scale

    x_scaled = np.clip(x_scaled, 0, original_w - 1)
    y_scaled = np.clip(y_scaled, 0, original_h - 1)

    # Error range (±1 pixel in padded image space, converted to original)
    pixel_error = 1
    x_range = (
        ((pred_norm[0] * TARGET_SIZE[0] - pixel_error) - pad_left) / scale,
        ((pred_norm[0] * TARGET_SIZE[0] + pixel_error) - pad_left) / scale
    )
    y_range = (
        ((pred_norm[1] * TARGET_SIZE[1] - pixel_error) - pad_top) / scale,
        ((pred_norm[1] * TARGET_SIZE[1] + pixel_error) - pad_top) / scale
    )

    x_range = (max(0, x_range[0]), min(original_w - 1, x_range[1]))
    y_range = (max(0, y_range[0]), min(original_h - 1, y_range[1]))

    return (x_scaled, y_scaled), x_range, y_range, img, scale

if __name__ == "__main__":
    model = load_model(MODEL_PATH, custom_objects={'ChannelAttention': ChannelAttention}, compile=False)

    coords, x_range, y_range, img, scale = predict_ball_location(INPUT_IMAGE_PATH, model)
    x_scaled, y_scaled = coords

    print(f"Predicted coords (x, y): ({x_scaled:.1f}, {y_scaled:.1f})")
    print(f"X Range: {x_range}")
    print(f"Y Range: {y_range}")

    x_int, y_int = int(round(x_scaled)), int(round(y_scaled))
    cross_size = 20
    color = (0, 255, 0)
    thickness = 3
    cv2.line(img, (x_int - cross_size, y_int), (x_int + cross_size, y_int), color, thickness)
    cv2.line(img, (x_int, y_int - cross_size), (x_int, y_int + cross_size), color, thickness)
    cv2.rectangle(img, (int(round(x_range[0])), int(round(y_range[0]))), (int(round(x_range[1])), int(round(y_range[1]))), (255, 255, 0), 2)

    output_path = os.path.join('static', 'scaled_prediction_with_pose.jpeg')
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")

    text = (
        f"Original size: {img.shape[1]}x{img.shape[0]}\n"
        f"Predicted coords (x, y): ({x_scaled:.1f}, {y_scaled:.1f})\n"
        f"X Range: {x_range[0]:.1f} to {x_range[1]:.1f}\n"
        f"Y Range: {y_range[0]:.1f} to {y_range[1]:.1f}\n"
        f"Scale used: {scale:.6f}\n"
    )
    with open('static/predicted_result_with_pose.txt', 'w') as f:
        f.write(text)
    print("Prediction data saved to static/predicted_result_with_pose.txt")
