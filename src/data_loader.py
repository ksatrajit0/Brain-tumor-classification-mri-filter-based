import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR

def generate_dataframe(data_dir):
    filepaths, labels = [], []
    for fold in os.listdir(data_dir):
        fold_path = os.path.join(data_dir, fold)
        for file in os.listdir(fold_path):
            filepaths.append(os.path.join(fold_path, file))
            labels.append(fold)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

def get_data_generators():
    train_df = generate_dataframe(TRAIN_DIR)
    test_df = generate_dataframe(TEST_DIR)
    train_df, valid_df = train_test_split(train_df, train_size=0.8, random_state=123, shuffle=True)

    gen = ImageDataGenerator()
    return (
        gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'),
        gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'),
        gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False),
        test_df
    )
