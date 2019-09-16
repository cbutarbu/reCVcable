import sys
import os
import numpy as np
import tensorflow as tf
from cv2 import cv2
from matplotlib import pyplot as plt

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import Dropout

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def getDatabase():
    data = "../data"
    classes = ['battery', 'disc', 'glass', 'metals', 'paper', 
        'plastic_jug_bottle', 'plastic_packaging', 'styrofoam']
    
    # Get absolute path of photos from data and add to folders array
    folders = []
    for classNames in classes:
        folder = os.path.join(data, classNames)
        image_paths = os.listdir(folder)
        absolute_path = [os.path.join(folder, image_path) 
            for image_path in image_paths]
        folders.append(absolute_path)

    # load images from absolute path of each photo
    images = []
    for paths in folders:
            class_images = [cv2.imread(path) for path in paths
                if cv2.imread(path) is not None]
            images.append(class_images)

    return images
    
# Build model
def createModel(train_data):
    classes = ['battery', 'disc', 'glass', 'metals', 'paper', 
        'plastic_jug_bottle', 'plastic_packaging', 'styrofoam']
    
    model = Sequential()
    # Add layers 
    model.add(Conv2D(32, (3,3), padding='same', input_shape=train_data.shape[1:], activation='relu', name='conv_1'))
    model.add(Conv2D(32, (3,3), activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv_3'))
    model.add(Conv2D(64, (3,3), activation='relu', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_2'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv_5'))
    model.add(Conv2D(128, (3,3), activation='relu', name='conv_6'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_3'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(len(classes), name='output'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # optimizer=RMSprop(lr=0.001)

    return model
  

# plot history of model
def plot_model_history(history, epochs):
    classesNum = 8
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
    plt.plot(np.arange(1, len(history['val_acc'])+1), history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='best')
  
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='best')
  
  
    plt.show()

# resize image to (256, 256)
def resizeImage(imageArr):
    resizedImages = []
    height, width = 256, 256
    for i, imgs in enumerate(imageArr):
        resizedImages.append([cv2.resize(img, (width, height), 
            interpolation = cv2.INTER_CUBIC) for img in imgs])
    return resizedImages

# Create labels and categories. Then, train images 
def trainImages(imagesArr, output):
    train_images = []
    val_images = []

    for imgs in imagesArr:
        # Partition training and testing data and save them into vars
        train, test = train_test_split(imgs, train_size = 0.8, test_size = 0.2)
        train_images.append(train)
        val_images.append(test)

    # Create labels for the images
    lenTrain = [len(imgs) for imgs in train_images]
    trainSum = np.sum(lenTrain)
    lenTest = [len(imgs) for imgs in val_images]
    valSum = np.sum(lenTest)
    NUM_OF_CLASSES = 8

    # Categorize each image for training and validating
    train_categories = np.zeros(trainSum, dtype = 'uint8')
    val_categories = np.zeros(valSum, dtype = 'uint8')
    for i in range(NUM_OF_CLASSES):
        if i is 0: 
            # Reserve 0 to represent first class, amount depending on first index
            # of lenTrain (number of photos for first class)
            train_categories[:lenTrain[i]] = i
            val_categories[:lenTest[i]] = i
        else:
            # Set the next slots, amount depending on the files in folder,
            # representing number of images for each class, to categorize a class
            # and stop until we have an index for all slots
            train_categories[np.sum(lenTrain[:i]):np.sum(lenTrain[:i+1])] = i
            val_categories[np.sum(lenTest[:i]):np.sum(lenTest[:i+1])] = i

    # Convert image to numpy
    temp_train = []
    temp_val = []

    for imgs in train_images:
        temp_train += imgs
    for imgs in val_images:
        temp_val += imgs

    train_images = np.array(temp_train)
    val_images = np.array(temp_val)

    train_data = train_images.astype('float32')
    val_data = val_images.astype('float32')
    train_labels = np_utils.to_categorical(train_categories, NUM_OF_CLASSES)
    val_labels = np_utils.to_categorical(val_categories, NUM_OF_CLASSES)

    # Shuffle data
    seed = 150
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(val_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)
    np.random.seed(seed)
    np.random.shuffle(val_labels)
    np.random.seed(seed)

    train_data = train_data[:1950]
    train_labels = train_labels[:1950]
    val_data = val_data[:490]
    val_labels = val_labels[:490]

    model = createModel(train_data)

    # Create generator
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.4,
        height_shift_range = 0.4,
        shear_range = 0.2,
        zoom_range = 0.3,
        horizontal_flip = True
    )

    val_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    # Flow images in batches
    batch_size = 2
    epochs = 50
    verboseAm = 2

    train_generator = train_datagen.flow(
        train_data,
        train_labels,
        batch_size = batch_size
    )

    val_generator = val_datagen.flow (
        val_data,
        val_labels,
        batch_size = batch_size
    )

    # Train the model
    model_train = model.fit_generator(
        generator = train_generator,
        steps_per_epoch=len(train_data)/batch_size,
        epochs=epochs,
        validation_steps = len(val_data)/batch_size,
        validation_data=val_generator,
        verbose=1
    )

    plot_model_history(model_train.history, epochs)
    model.save(output)

def predict_image (img, model):
    height, width = 256, 256
    img = cv2.resize(img, (width, heigh), interpolation = cv2.INTER_CUBIC)
    img = np.reshape(img, (1, width, heigh, 3))
    img = img/255
    pred = model.predict(img)
    class_num = np.argmax(pred)
    return class_num, np.max










        
        
                