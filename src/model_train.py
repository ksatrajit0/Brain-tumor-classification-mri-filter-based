from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf

def build_model(input_shape, num_classes):
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006)),
        Dropout(0.45),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
