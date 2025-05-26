# app/predictors/UNetHeatmap/predict.py
import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from app.utils.preprocess import resize_with_padding
from config import Config

model = load_model('app/predictors/UNetHeatmap/model.keras')

def predict_coordinates(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img.shape[:2]
    resized, scale, pad_left, pad_top = resize_with_padding(img, Config.TARGET_SIZE)
    input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    heatmap = model.predict(input_tensor)[0]

    # Find peak in heatmap
    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape[:2])

    # Map heatmap coords back to original
    heat_h, heat_w = heatmap.shape[:2]
    pred_x = (x_idx / heat_w * Config.TARGET_SIZE[0] - pad_left) / scale
    pred_y = (y_idx / heat_h * Config.TARGET_SIZE[1] - pad_top) / scale
    return round(pred_x, 1), round(pred_y, 1)
