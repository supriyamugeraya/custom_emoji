import numpy as np
import pickle
import cv2, os
from keras.api import optimizers
from keras.api.models import Sequential, load_model, save_model
from keras.api.layers import Dense, Dropout, Flatten, Input
from keras.api.layers import MaxPooling2D, Conv2D
from keras.api.utils import to_categorical
from keras.api.callbacks import ModelCheckpoint, TensorBoard
from keras.src.layers.normalization.batch_normalization import BatchNormalization
from keras.api import backend as K
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
	model.add(Conv2D(32, (5,5), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	# model.add(Dense(num_of_classes, activation='softmax'))
	num_of_classes = 6
	model.add(Dense(num_of_classes , activation='softmax'))
	sgd = optimizers.SGD(learning_rate=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras_1.keras"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	# from keras.api.utils import plot_model
	from keras.api.utils import plot_model
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
		print("len of train images ", len(train_images))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.uint8)
		print("len of train labels ", len(train_labels))

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.uint8)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.uint8)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))


	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)
	val_labels = to_categorical(val_labels)

	model, callbacks_list = cnn_model()
	tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
	callbacks_list.append(tensorboard)
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=50, callbacks=callbacks_list)
	# model = load_model('cnn_model_keras.h5')
	scores = model.evaluate(val_images, val_labels, verbose=1)
	save_model(model, 'cnn_model_keras_1.keras')
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

train()