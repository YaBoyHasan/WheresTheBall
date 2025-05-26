import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.predictors import ChannelAttention
from app.utils.preprocess import resize_with_padding
from config import Config

# Load model once at module level
model_path = 'app/predictors/VanillaCNN/model.keras'
model = load_model(model_path, custom_objects={'ChannelAttention': ChannelAttention})

def predict_coordinates(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    original_h, original_w = img.shape[:2]
    resized, scale, pad_left, pad_top = resize_with_padding(img, Config.TARGET_SIZE)
    input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    pred_norm = model.predict(input_tensor)[0]
    x_scaled = (pred_norm[0] * Config.TARGET_SIZE[0] - pad_left) / scale
    y_scaled = (pred_norm[1] * Config.TARGET_SIZE[1] - pad_top) / scale

    return round(x_scaled, 1), round(y_scaled, 1)
