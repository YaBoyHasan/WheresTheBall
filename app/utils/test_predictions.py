import os
import numpy as np
import cv2
from config import Config
from app.models.botbcomp import db, BotbComp

from app.predictors.VanillaCNN import predict as VanillaCNNPredict
from app.predictors.EfficientNet import predict as EfficientNetPredict

def getresponse():
    # Load data from DB
    comps = BotbComp.query.filter(BotbComp.JudgesX.isnot(None), BotbComp.JudgesY.isnot(None)).all()

    y_true = []
    y_preds = {
        "VanillaCNN": [],
        "EfficientNet": []
    }

    for comp in comps:
        img_path = os.path.join(Config.IMAGES_FOLDER, comp.ImageFileName)
        if not os.path.exists(img_path):
            continue

        y_true.append([comp.JudgesX, comp.JudgesY])

        y_preds["VanillaCNN"].append(VanillaCNNPredict.predict_coordinates(img_path))
        y_preds["EfficientNet"].append(EfficientNetPredict.predict_coordinates(img_path))

    y_true = np.array(y_true)

    def mean_distance(y_true, y_pred):
        return np.mean(np.linalg.norm(y_true - np.array(y_pred), axis=1))

    print("Model Performance (Lower is Better):")
    for name, preds in y_preds.items():
        score = mean_distance(y_true, preds)
        print(f"{name:20}: {score:.2f} average pixel distance")
