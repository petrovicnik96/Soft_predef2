import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Activation, \
    MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint

csvData = pd.read_csv('metadata/train/train_labels.csv')

img_width = 300
img_height = 300

X = []
Y = []

for i in tqdm(range(csvData.shape[0])):
    path = 'metadata/train/' + csvData['file_name'][i]
    img = image.load_img(path, target_size=(img_width, img_height, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

X = np.array(X)

for index, row in csvData.iterrows():
    print(index)
    if (row['weather'] == 'rain'):
        label = [1, 0, 0]
    elif (row['weather'] == 'sunrise'):
        label = [0, 1, 0]
    elif (row['weather'] == 'cloudy'):
        label = [0, 0, 1]
    elif (row['weather'] == 'shine'):
        label = [0, 0, 0]
    Y.append(label)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.15)

# AlexNet
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(300, 300, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(300 * 300 * 3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.1))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.2))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.summary()

# Compile the model


opt = Adam(learning_rate=0.000001)
# opt = SGD(lr=0.0001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
