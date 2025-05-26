import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from config import Config
from app.models.botbcomp import BotbComp, db  # assuming your SQLAlchemy setup

def augment(image, coord):
    h, w = image.shape[:2]
    coord = coord.copy()
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        coord[0] = w - coord[0]
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    coord[0] = np.clip(coord[0], 0, w-1)
    coord[1] = np.clip(coord[1], 0, h-1)
    return image, coord

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

def preprocess_and_save():
    data = db.session.query(BotbComp).filter(
        BotbComp.JudgesX != None,
        BotbComp.JudgesY != None,
        BotbComp.ImageFileName != None
    ).all()
    db.session.close()

    all_imgs, all_coords = [], []
    
    for row in tqdm(data, desc="Processing"):
        x, y = row.JudgesX, row.JudgesY
        img_path = os.path.join(Config.IMAGES_FOLDER, row.ImageFileName)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Missing image: {row.ImageFileName}")
            continue

        for _ in range(Config.NUM_AUGS):
            aug_img, aug_coord = augment(img.copy(), [x, y])
            resized, scale, pad_left, pad_top = resize_with_padding(aug_img, Config.TARGET_SIZE)

            norm_x = (aug_coord[0] * scale + pad_left) / Config.TARGET_SIZE[0]
            norm_y = (aug_coord[1] * scale + pad_top) / Config.TARGET_SIZE[1]

            all_imgs.append(resized)
            all_coords.append([norm_x, norm_y])

    X = np.array(all_imgs, dtype=np.float32) / 255.0
    y = np.array(all_coords, dtype=np.float32)

    np.savez_compressed(Config.PROCESSED_DATA_PATH, X=X, y=y)
    print(f"\n✅ Saved {len(X)} samples to {Config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_and_save()
