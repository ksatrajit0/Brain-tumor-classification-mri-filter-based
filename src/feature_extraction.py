import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

def extract_features(img_dir, labels_dict, weights_path='/your_folder_name/efficientb3_braintumor_weights.h5'):
    base_model = EfficientNetB3(include_top=False, input_shape=(256, 256, 3))
    if weights_path:
        base_model.load_weights(weights_path, by_name=True)
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    
    all_features = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, file)
                img = image.load_img(full_path, target_size=(256, 256))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                processed = preprocess_input(img_array)
                features = model.predict(processed).flatten()
                label = labels_dict[os.path.basename(root)]
                row = [os.path.basename(root) + "_" + file] + list(features) + [label]
                all_features.append(row)
    return pd.DataFrame(all_features)
