# app/predictors/heatmap_utils.py

import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D

def create_heatmap_targets(coordinates, shape=(28, 28), sigma=1):
    """
    Generates heatmap targets from (x, y) coordinates.
    """
    heatmaps = []
    h, w = shape
    for coord in coordinates:
        x, y = coord
        heatmap = np.zeros((h, w), dtype=np.float32)
        if 0 <= x < 1 and 0 <= y < 1:
            cx, cy = int(x * w), int(y * h)
            heatmap = draw_gaussian(heatmap, (cx, cy), sigma)
        heatmaps.append(heatmap[..., np.newaxis])
    return np.array(heatmaps)

def draw_gaussian(heatmap, center, sigma):
    """
    Draw a 2D Gaussian on a heatmap.
    """
    x, y = center
    height, width = heatmap.shape[:2]

    size = int(3 * sigma)
    x0 = max(0, x - size)
    x1 = min(width, x + size + 1)
    y0 = max(0, y - size)
    y1 = min(height, y + size + 1)

    if x1 <= x0 or y1 <= y0:
        return heatmap

    xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)
    return heatmap

def FastHeatmapNet(input_shape=(224, 224, 3), heatmap_shape=(28, 28)):
    """
    MobileNetV2 backbone + 3x upsampling conv layers to output heatmap.
    """
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = base.output  # e.g. (7,7,1280)
    x = UpSampling2D(2)(x)         # (14,14)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(2)(x)         # (28,28)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)  # Final heatmap output
    model = Model(inputs=base.input, outputs=x)
    return model
