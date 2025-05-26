# app/predictors/UNetHeatmap/train.py
import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from config import Config
from app.predictors.heatmap_utils import create_heatmap_targets, UNet

def run_training():
    data = np.load(Config.PROCESSED_DATA_PATH)
    X, y = data['X'], data['y']
    heatmaps = create_heatmap_targets(y, shape=Config.HEATMAP_SHAPE)
    X_train, X_test, y_train, y_test = train_test_split(X, heatmaps, test_size=0.2)

    model = UNet(input_shape=Config.TARGET_SHAPE)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)
    model.save('app/predictors/UNetHeatmap/model.keras')
