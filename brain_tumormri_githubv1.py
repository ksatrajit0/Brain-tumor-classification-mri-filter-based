# -*- coding: utf-8 -*-
"""brain_tumorMRI_GitHubv1.ipynb

# **BRAIN TUMOR MRI MULTI-CLASS CLASSIFICATION USING DEEP LEARNING**
- authored by Satrajit Kar, MESE@JU'25
- [Brain Tumor MRI Dataset (SARTAJ+Br35H+figshare)]

Import files and modules
"""

from google.colab import files
print("Upload your kaggle.json")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
!unzip \*.zip  && rm *.zip

import shutil
import os
dir_train = '/content/Training'
dir_test = '/content/Testing'
new_dir = '/content/brain-tumor-classification-mri'
os.makedirs(new_dir, exist_ok=True)
shutil.move(dir_train, new_dir)
shutil.move(dir_test, new_dir)

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

"""Data Preprocessing
- Sample Image in Each Stage of Data Preprocessing
"""

import numpy as np
import cv2
import os
import imutils
import matplotlib.pyplot as plt

def crop_img(img, save_path):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    fig, axs = plt.subplots(1, 7, figsize=(20, 20))

    # Stage 0: Show Original Image
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')

    # Stage 1: Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(save_path + '/grayscale1.png', gray)
    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title('Grayscale Conversion')

    # Stage 2: Gaussian Blurring
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(save_path + '/gaussian_blurred2.png', gray)
    axs[2].imshow(gray, cmap='gray')
    axs[2].set_title('Gaussian Blurring')

    # Stage 3: Thresholding
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(save_path + '/thresholded3.png', thresh)
    axs[3].imshow(thresh, cmap='gray')
    axs[3].set_title('Thresholding')

    # Stage 4: Erosion and Dilation
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imwrite(save_path + '/eroded_dilated4.png', thresh)
    axs[4].imshow(thresh, cmap='gray')
    axs[4].set_title('Erosion and Dilation')

    # Stage 5: Contour Detection
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Stage 6: Extreme Point Detection and Image Cropping
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    cv2.imwrite(save_path + '/cropped5.png', new_img)
    axs[5].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    axs[5].set_title('Cropped Image')

    # Stage 7: Resizing
    new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
    cv2.imwrite(save_path + '/resized6.png', new_img)
    axs[6].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    axs[6].set_title('Resized Image')

    # Draw arrows
    for i in range(6):
        axs[i].annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                        xycoords=axs[i].transAxes, textcoords=axs[i+1].transAxes,
                        arrowprops=dict(arrowstyle="<-", linewidth=2, color = 'black'))
        axs[i].axis('off')
    axs[6].axis('off')

    #Adjust spacing between subplots

    plt.subplots_adjust(wspace=0.5)

    plt.show()

    return new_img

if __name__ == "__main__":
    training = "/content/brain-tumor-classification-mri/Training"
    IMG_SIZE = 256

    dir = 'glioma'  # select the class
    path = os.path.join(training,dir)
    image_dir = os.listdir(path)
    img = image_dir[0]  # select the first image
    image = cv2.imread(os.path.join(path,img))
    img_save_path = '/content/data_preprocessing/' + dir + '/' + img.split('.')[0]  # create a new folder for the image
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    cv2.imwrite(img_save_path + '/original.png', image)  # save the original image
    new_img = crop_img(image, img_save_path)
    new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
    cv2.imwrite(img_save_path + '/Resized.png', new_img)

"""Data Cleaning"""

import numpy as np
from tqdm import tqdm
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
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

	return new_img

if __name__ == "__main__":
	training = "/content/brain-tumor-classification-mri/Training"
	testing = "/content/brain-tumor-classification-mri/Testing"
	training_dir = os.listdir(training)
	testing_dir = os.listdir(testing)
	IMG_SIZE = 256

	for dir in training_dir:
		save_path = '/content/cleaned/Training/'+ dir
		path = os.path.join(training,dir)
		image_dir = os.listdir(path)
		for img in image_dir:
			image = cv2.imread(os.path.join(path,img))
			new_img = crop_img(image)
			new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			cv2.imwrite(save_path+'/'+img, new_img)

	for dir in testing_dir:
		save_path = '/content/cleaned/Testing/'+ dir
		path = os.path.join(testing,dir)
		image_dir = os.listdir(path)
		for img in image_dir:
			image = cv2.imread(os.path.join(path,img))
			new_img = crop_img(image)
			new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			cv2.imwrite(save_path+'/'+img, new_img)

# Generate data paths with labels
train_data_dir = '/content/cleaned/Training'
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

train_df

# Generate data paths with labels
test_data_dir = '/content/cleaned/Testing'
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

ts_df

"""train, validation and test split"""

# valid and test dataframe
train_df, valid_df = train_test_split(train_df,  train_size= 0.8, shuffle= True, random_state= 123)

# cropped image size
batch_size = 16
img_size = (256,256)
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

"""EfficientNetB3 Model"""

from tensorflow.keras.applications import EfficientNetB3

# Create Model Structure
img_size = (256, 256)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer


base_model = tf.keras.applications.EfficientNetB3(include_top= False, weights= 'imagenet', input_shape= img_shape, pooling= 'max')

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

history = model.fit(x= train_gen, epochs= 30, verbose= 1, validation_data= valid_gen,
                    validation_steps= None, shuffle= False)

model.save_weights("/content/efficientnetb3_braintumor_weights.h5")

import pandas as pd

# Create a DataFrame from the history dictionary
history_df = pd.DataFrame(history.history)

# Use the DataFrame's 'to_string' method to print the DataFrame in a tabular format
print(history_df.to_string())

#Plot the model's performance

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

# Test its accuracy

ts_length = len(ts_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps= test_steps, verbose= 1)
valid_score = model.evaluate(valid_gen, steps= test_steps, verbose= 1)
test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)
plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()


print(classification_report(test_gen.classes, y_pred, target_names= classes))

!pip install pydot graphviz

# Import necessary modules
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from IPython.display import Image

model_visual = models.Model(inputs=model.input, outputs=model.output)

# Save model architecture to a file
plot_model(model_visual, show_dtype=True, to_file='efficientnetb3_model_architecture.png', show_shapes=True)

# Display model architecture in the notebook
Image(retina=True, filename='efficientnetb3_model_architecture.png')

!pip install visualkeras

from visualkeras import layered_view
# Visualize the model
layered_view(model, legend=True, max_xy=300)

"""Feature Extraction using Pretrained Model Weights"""

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
base_model_efnetb3 = EfficientNetB3(include_top=False, input_shape=(256, 256, 3))
base_model_efnetb3.load_weights('/content/efficientnetb3_braintumor_weights.h5', by_name=True)

x = base_model_efnetb3.output
x = GlobalAveragePooling2D()(x)
feature_extraction_model = Model(inputs=base_model_efnetb3.input, outputs=x)

labels = {'glioma':1,'notumor':0,'meningioma':2,'pituitary':3}
training_global_average_pool_2d_efnetb3 = []
testing_global_average_pool_2d_efnetb3 = []

main_folder = '/content/brain-tumor-classification-mri/Training'
total_files=1
subfiles=1
# Iterate through subfolders and print subfolder name and file names
for root, dirs, files in os.walk(main_folder):
  print(total_files)
  for file in files:
    if file.endswith(('jpg', 'jpeg', 'png')):
      d=[]
      file_path = os.path.join(root, file)  # Check for image file formats
      img = image.load_img(file_path, target_size=(256, 256))
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

training_df

main_folder = '/content/brain-tumor-classification-mri/Testing'
total_files=1
subfiles=1
# Iterate through subfolders and print subfolder name and file names
for root, dirs, files in os.walk(main_folder):
  print(total_files)
  for file in files:
    if file.endswith(('jpg', 'jpeg', 'png')):
      d=[]
      file_path = os.path.join(root, file)  # Check for image file formats
      img = image.load_img(file_path, target_size=(256, 256))
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

testing_df

from google.colab import files

# Save Testing DataFrame to a CSV file
testing_df.to_csv('testing_global_average_pool_2d_efnetb3_trained_weights_new.csv', index=False)

# Save Training DataFrame to a CSV file
training_df.to_csv('training_global_average_pool_2d_efnetb3_trained_weights_new.csv', index=False)

"""Finding Best Number of Reduced Features to be used for Mutual Information"""

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
df = pd.read_csv('/content/training_global_average_pool_2d_efnetb3_trained_weights_new.csv')

# Separate features and target
X_train = df.iloc[:, 1:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/testing_global_average_pool_2d_efnetb3_trained_weights_new.csv')

X_test = df_test.iloc[:, 1:-1]
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

"""SVM Classifier (Polynomial Kernel), without Mutual Information"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,recall_score,cohen_kappa_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training dataset
df = pd.read_csv('/content/drive/MyDrive/BrainTumor/Final/training_global_average_pool_2d_efnetb3_trained_weights_new.csv')

# Separate features and target
X_train = df.iloc[:, 1:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Train the model
clf.fit(X_train, y_train)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/drive/MyDrive/BrainTumor/Final/testing_global_average_pool_2d_efnetb3_trained_weights_new.csv')

X_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:, -1]

X_test = scaler.transform(X_test)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Define the class names
class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']

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

"""SVM Classifier (Polynomial Kernel), with Mutual Information"""

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
df = pd.read_csv('/content/drive/MyDrive/BrainTumor/Final/training_global_average_pool_2d_efnetb3_trained_weights_new.csv')

# Separate features and target
X_train = df.iloc[:, 1:-1]
y_train = df.iloc[:, -1]

# Initialize the MinMaxScaler and apply it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Apply Mutual Information for feature selection
selector = SelectKBest(mutual_info_classif, k=1140)
X_train = selector.fit_transform(X_train, y_train)

# Initialize the SVM Classifier
clf = SVC(kernel='poly', gamma = 0.5, C = 0.1, random_state=7)

# Train the model
clf.fit(X_train, y_train)

# Load the unseen testing dataset
df_test = pd.read_csv('/content/drive/MyDrive/BrainTumor/Final/testing_global_average_pool_2d_efnetb3_trained_weights_new.csv')

X_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:, -1]

X_test = scaler.transform(X_test)
X_test = selector.transform(X_test)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Define the class names
class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']

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

precision_per_class = precision_score(y_test, y_pred, average = None)
recall_per_class = recall_score(y_test, y_pred, average=None)
f1_per_class = f1_score(y_test, y_pred,average=None)

class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
for i in range (len(precision_per_class)):
  print(f'Class: {class_names[i]} = Precision: {(precision_per_class[i]) * 100}, Recall = {(recall_per_class[i]) * 100}')

class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

for i in range(len(class_names)):
    accuracy = cm[i,i] / sum(cm[i,:])
    print(f'Class: {class_names[i]} = Accuracy: {accuracy * 100}, F1_Score = {(f1_per_class[i])*100}')



"""Visualization of Results"""

import matplotlib.pyplot as plt
import numpy as np

class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Calculate metrics
precision_per_class = np.round(precision_score(y_test, y_pred, average=None), 2) * 100
recall_per_class = np.round(recall_score(y_test, y_pred, average=None), 2) * 100
f1_per_class = np.round(f1_score(y_test, y_pred,average=None), 2) * 100

# Calculate accuracy for each class
cm = confusion_matrix(y_test, y_pred)
accuracy_per_class = np.round([cm[i,i] / sum(cm[i,:]) for i in range(len(class_names))], 2) * 100

# Prepare data for bar plot
labels = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
x = np.arange(len(labels))  # the label locations
width = 0.23  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - 3*width/2, precision_per_class, width, label='Precision')
rects2 = ax.bar(x - width/2, recall_per_class, width, label='Recall')
rects3 = ax.bar(x + width/2, f1_per_class, width, label='F1 Score')
rects4 = ax.bar(x + 3*width/2, accuracy_per_class, width, label='Accuracy')

# Function to annotate bars with their respective values
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call autolabel function for each set of bars
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Scores')
ax.set_title('Performance Scores by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')  # Show legend beside the graph

# Increase the bottom margin to prevent x-labels from being cut-off
plt.subplots_adjust(bottom=0.15)

# Set y-axis limits to show values from 0.9 to 1.0
plt.ylim(90, 100)

plt.show()
