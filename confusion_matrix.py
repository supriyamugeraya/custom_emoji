from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.api.models import load_model
from sklearn.model_selection import StratifiedKFold
import dlib, os
import cv2
import numpy as np
import pickle
from time import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CNN_MODEL = 'cnn_best_model.keras'
SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"

def lr_schedule(epoch, lr):
    return lr * 0.95 if epoch > 5 else lr 

def get_image_size():
    img = cv2.imread('new_dataset/0/1.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(os.listdir('new_dataset/'))

image_x, image_y = get_image_size()


cnn_model = load_model(CNN_MODEL)
shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector = dlib.get_frontal_face_detector()


with open("train_images", "rb") as f:
    train_images = np.array(pickle.load(f))
with open("train_labels", "rb") as f:
    train_labels = np.array(pickle.load(f), dtype=np.uint8)

train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
test_labels=train_labels
train_labels = to_categorical(train_labels)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
for train_idx, val_idx in kfold.split(train_images, np.argmax(train_labels, axis=1)):
    tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
    checkpoint = ModelCheckpoint("cnn_best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    callbacks_list = [tensorboard, checkpoint, early_stop, lr_scheduler]

    x_train, x_val = train_images[train_idx], train_images[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]

    y_pred=cnn_model.predict(x_val)
    new_pred_class=[]
    for pred_probab in y_pred:
    
        pred_class = list(pred_probab).index(max(pred_probab))
        new_pred_class.append(pred_class)
    y_val=np.argmax(y_val,axis=1)
    
    cm=confusion_matrix(y_val,new_pred_class)
    plt.figure(figsize=(8,6))
    class_names=['Happy','Sad','Wink','Pout','Shock','Angry','Neutral']
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",xticklabels=class_names,yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    