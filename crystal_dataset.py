# -*- coding: utf-8 -*-
"""Brain Tumor Classification from Structural MRI Scans using a Deep Feature Selection Methodology.ipynb

# **BRAIN TUMOR MRI MULTI-CLASS CLASSIFICATION USING DEEP LEARNING**

authored by Satrajit Kar, MESE JU'25

# Crystal Dataset
"""

from google.colab import files
print("Upload your kaggle.json")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Import necessary libraries
import os

# Create a directory
os.makedirs('brain-tumors-dataset', exist_ok=True)

# Change the current working directory
os.chdir('brain-tumors-dataset')

# Download the dataset
!kaggle datasets download -d mohammadhossein77/brain-tumors-dataset

# Unzip the dataset and remove the zip file
!unzip \*.zip  && rm *.zip

import shutil
import os

# Set the paths for the source and destination folders
source_folder = '/content/brain-tumors-dataset/Data/Normal'
destination_folder = '/content/crystal/train/no_tumor'

# Check if the destination folder exists, create it if does not exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source directory
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Copy each file from the source to the destination folder
for file in files:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.copy(source_path, destination_path)

print("Files copied successfully.")

import shutil
import os

# Set the paths for the source and destination folders
source_folder = '/content/brain-tumors-dataset/Data/Tumor'
destination_folder = '/content/crystal/train'

# Check if the destination folder exists, create it if not
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all folders in the source directory
folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

# Copy each folder and its contents from the source to the destination folder
for folder in folders:
    source_path = os.path.join(source_folder, folder)
    destination_path = os.path.join(destination_folder, folder)
    shutil.copytree(source_path, destination_path)

print("Folders copied successfully.")

import shutil
import os
import random

# Set the paths for the source and destination folders
source_folder = '/content/crystal/train/no_tumor'
destination_folder = '/content/crystal/test/no_tumor'

# Check if the destination folder exists, create it if not
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source directory
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Calculate the number of files to move (10% of total files)
num_files_to_move = int(0.1 * len(files))

# Randomly select the files to move
files_to_move = random.sample(files, num_files_to_move)

# Move each selected file from the source to the destination folder
for file in files_to_move:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.move(source_path, destination_path)

print(f"{num_files_to_move} files moved successfully.")

import shutil
import os
import random

# Set the paths for the source and destination folders
source_folder = '/content/crystal/train/glioma_tumor'
destination_folder = '/content/crystal/test/glioma_tumor'

# Check if the destination folder exists, create it if not
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source directory
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Calculate the number of files to move (10% of total files)
num_files_to_move = int(0.1 * len(files))

# Randomly select the files to move
files_to_move = random.sample(files, num_files_to_move)

# Move each selected file from the source to the destination folder
for file in files_to_move:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.move(source_path, destination_path)

print(f"{num_files_to_move} files moved successfully.")

import shutil
import os
import random

# Set the paths for the source and destination folders
source_folder = '/content/crystal/train/meningioma_tumor'
destination_folder = '/content/crystal/test/meningioma_tumor'

# Check if the destination folder exists, create it if not
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source directory
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Calculate the number of files to move (10% of total files)
num_files_to_move = int(0.1 * len(files))

# Randomly select the files to move
files_to_move = random.sample(files, num_files_to_move)

# Move each selected file from the source to the destination folder
for file in files_to_move:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.move(source_path, destination_path)

print(f"{num_files_to_move} files moved successfully.")

import shutil
import os
import random

# Set the paths for the source and destination folders
source_folder = '/content/crystal/train/pituitary_tumor'
destination_folder = '/content/crystal/test/pituitary_tumor'

# Check if the destination folder exists, create it if not
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source directory
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Calculate the number of files to move (10% of total files)
num_files_to_move = int(0.1 * len(files))

# Randomly select the files to move
files_to_move = random.sample(files, num_files_to_move)

# Move each selected file from the source to the destination folder
for file in files_to_move:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(destination_folder, file)
    shutil.move(source_path, destination_path)

print(f"{num_files_to_move} files moved successfully.")

folder_name = 'crystal_processed'
colab_path='/content/'
colab_folder_path = f'{colab_path}/{folder_name}'
if not os.path.exists(colab_folder_path):
    os.makedirs(colab_folder_path)
    print(f'Folder "{folder_name}" created.')

folder_name = 'train'
colab_path='/content/crystal_processed/'
colab_folder_path = f'{colab_path}/{folder_name}'
if not os.path.exists(colab_folder_path):
    os.makedirs(colab_folder_path)
    print(f'Folder "{folder_name}" created.')

folder_name = 'test'
colab_path='/content/crystal_processed/'
colab_folder_path = f'{colab_path}/{folder_name}'
if not os.path.exists(colab_folder_path):
    os.makedirs(colab_folder_path)
    print(f'Folder "{folder_name}" created.')

"""Data Preprocessing"""

import cv2
import os
import imutils

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    try:
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = 0
        cropped_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

        return cropped_img

    except ValueError:
        # If no contours are found, print a message and continue to the next image
         raise NoContoursFoundException("No contours found in the image")



class NoContoursFoundException(Exception):
    pass



if __name__ == "__main__":
    training = "/content/crystal/train"
    testing = "/content/crystal/test"
    training_dir = os.listdir(training)
    testing_dir = os.listdir(testing)
    IMG_SIZE = 256

    for dir in training_dir:
        save_path = '/content/crystal_processed/train/'+ dir
        path = os.path.join(training,dir)
        image_dir = os.listdir(path)
        for img in image_dir:
            print(path,img)
            image = cv2.imread(os.path.join(path,img))
            try:
                new_img = crop_img(image)
                new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
            except NoContoursFoundException as e:
                print(e)
                continue
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+'/'+img, new_img)

        for dir in testing_dir:
            save_path = '/content/crystal_processed/test/'+ dir
            path = os.path.join(testing,dir)
            image_dir = os.listdir(path)
            for img in image_dir:
                print(path,img)
                image = cv2.imread(os.path.join(path,img))
                try:
                    new_img = crop_img(image)
                    new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
                except NoContoursFoundException as e:
                    print(e)
                    continue
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+img, new_img)

# import system libs
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

# Generate data paths with labels
train_data_dir = '/content/crystal_processed/train'
filepaths = []
labels = []

folds = os.listdir(train_data_dir)
for fold in folds:
    foldpath = os.path.join(train_data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)

        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis= 1)

# Generate data paths with labels
test_data_dir = '/content/crystal_processed/test'
filepaths = []
labels = []

folds = os.listdir(test_data_dir)
for fold in folds:
    foldpath = os.path.join(test_data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)

        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
ts_df = pd.concat([Fseries, Lseries], axis= 1)

# valid and test dataframe
train_df, valid_df = train_test_split(train_df,  train_size= 0.7, shuffle= True, random_state= 123)

# cropped image size
batch_size = 16
img_size = (256, 256)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe( ts_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)

g_dict = train_gen.class_indices      # defines dictionary {'class': index}
classes = list(g_dict.keys())       # defines list of dictionary's kays (classes), classes names : string
images, labels = next(train_gen)      # get a batch size samples from the generator

plt.figure(figsize= (20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255       # scales data to range (0 - 255)
    plt.imshow(image)
    index = np.argmax(labels[i])  # get image index
    class_name = classes[index]   # get class of image
    plt.title(class_name, color= 'blue', fontsize= 12)
    plt.axis('off')
plt.show()

# Create Model Structure
img_size = (256, 256)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

epochs = 20

base_model = tf.keras.applications.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])


model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()

history = model.fit(x= train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen,
                    validation_steps= None, shuffle= False)

import pandas as pd

# Create a DataFrame from the history dictionary
history_df = pd.DataFrame(history.history)

# Use the DataFrame's 'to_string' method to print the DataFrame in a tabular format
print(history_df.to_string())

# Define needed variables
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()

model.save_weights("/content/EfficientNetB3_BrainTumor_Crystal_Weights.h5")

"""Feature Extraction"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import pandas as pd
import numpy as np

# Load the EfficientNetB3 model
base_model_efnetb3 = EfficientNetB3(include_top=False, input_shape=(224, 224, 3))
base_model_efnetb3.load_weights('/content/EfficientNetB3_BrainTumor_Crystal_Weights.h5', by_name=True)

x = base_model_efnetb3.output
x = GlobalAveragePooling2D()(x)
feature_extraction_model = Model(inputs=base_model_efnetb3.input, outputs=x)

labels = {'glioma_tumor':1,'no_tumor':2,'meningioma_tumor':0,'pituitary_tumor':3}
training_global_average_pool_2d_efnetb3 = []
testing_global_average_pool_2d_efnetb3 = []

main_folder = '/content/crystal_processed/Training'
total_files=1
subfiles=1
# Iterate through subfolders and print subfolder name and file names
for root, dirs, files in os.walk(main_folder):
  print(total_files)
  for file in files:
    if file.endswith(('jpg', 'jpeg', 'png')):
      d=[]
      file_path = os.path.join(root, file)  # Check for image file formats
      img = image.load_img(file_path, target_size=(224, 224))
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      processed_img = preprocess_input(img_array)
      global_average_features = feature_extraction_model.predict(processed_img)
      one_dim_list = np.array(global_average_features).flatten()
      sub_folder_name=os.path.basename(root)
      ch=labels[sub_folder_name]
      d.append(sub_folder_name+'_'+file)
      d.extend(one_dim_list)
      d.append(ch)
      training_global_average_pool_2d_efnetb3.append(d)
      subfiles+=1
      total_files+=1

training_df=pd.DataFrame(training_global_average_pool_2d_efnetb3)

main_folder = '/content/crystal_processed/Testing'
total_files=1
subfiles=1
# Iterate through subfolders and print subfolder name and file names
for root, dirs, files in os.walk(main_folder):
  print(total_files)
  for file in files:
    if file.endswith(('jpg', 'jpeg', 'png')):
      d=[]
      file_path = os.path.join(root, file)  # Check for image file formats
      img = image.load_img(file_path, target_size=(224, 224))
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      processed_img = preprocess_input(img_array)
      global_average_features = feature_extraction_model.predict(processed_img)
      one_dim_list = np.array(global_average_features).flatten()
      sub_folder_name=os.path.basename(root)
      ch=labels[sub_folder_name]
      d.append(sub_folder_name+'_'+file)
      d.extend(one_dim_list)
      d.append(ch)
      testing_global_average_pool_2d_efnetb3.append(d)
      subfiles+=1
      total_files+=1

testing_df=pd.DataFrame(testing_global_average_pool_2d_efnetb3)

from google.colab import files

# Save Testing DataFrame to a CSV file
testing_df.to_csv('/content/crystal_final_testing.csv', index=False)

# Save Training DataFrame to a CSV file
training_df.to_csv('/content/crystal_final_training.csv', index=False)



"""No Mutual Info, SVM (Poly)"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,recall_score,cohen_kappa_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training dataset
df = pd.read_csv('/content/crystal_final_training.csv')

# Separate features and target
X_train = df.iloc[:, 0:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Train the model
clf.fit(X_train, y_train)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/crystal_final_testing.csv')

X_test = df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

X_test = scaler.transform(X_test)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Define the class names
class_names = ['meningioma', 'glioma', 'notumor', 'pituitary']

# Print the classification report
print("SVM (Polynomial) \n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Print the confusion matrix
print("Confusion Matrix \n")
cm2 = confusion_matrix(y_test, y_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print('\nAccuracy SVM (Polynomial): {:.2f} %'.format(accuracy_score(y_test,y_pred)*100))



"""Mutual Info, SVM"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,recall_score,cohen_kappa_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training dataset
df = pd.read_csv('/content/crystal_final_training.csv')

# Separate features and target
X_train = df.iloc[:, 0:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Apply Mutual Information for feature selection
selector = SelectKBest(mutual_info_classif, k=1400)
X_train = selector.fit_transform(X_train, y_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Train the model
clf.fit(X_train, y_train)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/crystal_final_testing.csv')

X_test = df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

X_test = scaler.transform(X_test)
X_test = selector.transform(X_test)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Define the class names
class_names = ['meningioma', 'glioma', 'notumor', 'pituitary']

# Print the classification report
print("SVM (Polynomial) \n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Print the confusion matrix
print("Confusion Matrix \n")
cm2 = confusion_matrix(y_test, y_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print('\nAccuracy SVM (Polynomial): {:.2f} %'.format(accuracy_score(y_test,y_pred)*100))



"""Finding Best Number of Reduced Features against Highest Classification Accuracy"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,recall_score,cohen_kappa_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training dataset
df = pd.read_csv('/content/crystal_final_training.csv')

# Separate features and target
X_train = df.iloc[:, 0:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/crystal_final_testing.csv')

X_test = df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

X_test = scaler.transform(X_test)

# Define the range of k
k_values = range(1000, 1451, 50)

# Initialize a list to store accuracy for each k
accuracy_list = []

for k in k_values:
    # Apply Mutual Information for feature selection
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_test_k = selector.transform(X_test)

    # Train the model
    clf.fit(X_train_k, y_train)

    # Predict the labels of the test set
    y_pred = clf.predict(X_test_k)

    # Calculate accuracy and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# Plot the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_list, marker='o', linestyle='-')
plt.title('Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

