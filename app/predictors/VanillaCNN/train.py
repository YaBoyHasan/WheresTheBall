# app/predictors/VanillaCNN/train.py
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from config import Config

def run_training():
    data = np.load(Config.PROCESSED_DATA_PATH)
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = models.Sequential([
        layers.Input(shape=Config.TARGET_SHAPE),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',    # watch validation loss
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[early_stop]
    )

    model.save('app/predictors/VanillaCNN/model.keras')