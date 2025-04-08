import numpy as np
import pickle
import cv2, os
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import optimizers, models
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('new_dataset/0/1.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(os.listdir('new_dataset/'))

image_x, image_y = get_image_size()

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Input((image_x, image_y, 1)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def lr_schedule(epoch, lr):
    return lr * 0.95 if epoch > 5 else lr

def stratified_train(train_new_data_only=False):
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.uint8)

    # Define the index limit for old data if only training on new data
    old_data_limit = 300  # Modify this index based on your dataset

    if train_new_data_only:
        # Use only the new data
        train_images = train_images[old_data_limit:]
        train_labels = train_labels[old_data_limit:]

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
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

    # Load existing model or create a new one
    if os.path.exists("cnn_final_model.keras"):
        model = load_model("cnn_final_model.keras")
    else:
        model = cnn_model()

    for train_idx, val_idx in kfold.split(train_images, np.argmax(train_labels, axis=1)):
        tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
        checkpoint = ModelCheckpoint("cnn_best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)

        callbacks_list = [tensorboard, checkpoint, early_stop, lr_scheduler]

        x_train, x_val = train_images[train_idx], train_images[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        model.fit(datagen.flow(x_train, y_train, batch_size=50), 
                  validation_data=(x_val, y_val), 
                  epochs=20, 
                  callbacks=callbacks_list)

        scores = model.evaluate(x_val, y_val, verbose=1)
        cvscores.append(scores[1] * 100)

    save_model(model, 'cnn_final_model.keras')
    print("Stratified CNN Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Call the function with train_new_data_only=False to retrain on the full dataset
# or train_new_data_only=True to train only on the new data
stratified_train(train_new_data_only=True)  # Change to False if you want to retrain on all data
