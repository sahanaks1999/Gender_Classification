from keras.applications import VGG16
import cv2
import time
import subprocess
import numpy as np
#import plotConfusionMatrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K


folders = subprocess.check_output('ls /home/sahanaks/Desktop/gender_classification/dataset', shell=True)#used to run  arguments and return strings
folders = folders.decode().split('\n')
folders.pop()

X = []#input vector
Y = []#target vector

i = 0   
for folder in folders:
    images = subprocess.check_output("ls /home/sahanaks/Desktop/gender_classification/dataset/" + str(folder), shell=True)
    images = images.decode().split('\n')
    images.pop()
    for image in images:
        img = cv2.imread('/home/sahanaks/Desktop/gender_classification/dataset/' + str(folder) + '/' + str(image))
        img = cv2.resize(img,(64,64))
        X.append(img)
        Y.append(i)
    i += 1

X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X /= 255

Y = keras.utils.to_categorical(Y, 2)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(xtrain)
#Load the VGG model
image_size = 64
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
    
from keras import models
from keras import layers
from keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print(K.eval(model.optimizer.lr))
early_stopping = EarlyStopping(patience=5, verbose=1)
model_checkpoint = ModelCheckpoint("/home/sahanaks/gender.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.00001, verbose=1)
model.summary()

history = model.fit(xtrain, ytrain, shuffle=True, batch_size=35,validation_split=0.1, epochs=1000, callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=1)
model.save('gender.model')
stop=time.time()
dur=stop-start
print('execution time:', str(dur/60) )
print(model.summary())
ypred = model.predict(xtest)
score = model.evaluate(xtest, ytest)
print('Test Accuracy = ' + str(score[1]*100))
print('Test Loss = ' + str(score[0]))



