# app/predictors/VanillaCNN/predict.py
import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from app.utils.preprocess import resize_with_padding
from config import Config

model = load_model('app/predictors/VanillaCNN/model.keras')

def predict_coordinates(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img.shape[:2]
    resized, scale, pad_left, pad_top = resize_with_padding(img, Config.TARGET_SIZE)
    input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    pred = model.predict(input_tensor)[0]
    x = (pred[0] * Config.TARGET_SIZE[0] - pad_left) / scale
    y = (pred[1] * Config.TARGET_SIZE[1] - pad_top) / scale
    return round(x, 1), round(y, 1)
