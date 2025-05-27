# app/predictors/EfficientNet/train.py
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from config import Config

def run_training():
    data = np.load(Config.PROCESSED_DATA_PATH)
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    base = EfficientNetB0(input_shape=Config.TARGET_SHAPE, include_top=False, weights='imagenet')
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(2, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop]
    )
    model.save('app/predictors/EfficientNet/model.keras')