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

# for i in tqdm(range(csvData.shape[0])):
# y = csvData.drop(columns=['file_name', 'weather'])[0:csvData.shape[0]]

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

# 1st layer
model.add(Conv2D(filters=96, input_shape=(300, 300, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
#max pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
#max pooling
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

#optimisers
opt = Adam(learning_rate=0.000001)
# opt = SGD(lr=0.0001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

epochs = 40

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_accuracy', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current > 0.84:
            print(current)
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

callback = EarlyStoppingByLossVal(monitor='val_accuracy', value=0.00001, verbose=1)
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy')

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print(type(X_train))
print(type(y_train))
print(type(y_test))

print(type(X_test))

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=1)

# model.load_weights('.mdl_wts.hdf5')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)

print("* VISUALIZING ACCURACY AND LOSS")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('ACCURACY')
# plt.savefig('accuracy.png')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper left')
plt.title('LOSS')
# plt.savefig('Loss.png')
plt.show()

# TESTING
miss = 0
hit = 0
test_data = pd.read_csv('metadata-test/test/test_labels.csv')

Y_test_data = []
for index, row in test_data.iterrows():
    print(index)
    if (row['weather'] == 'rain'):
        label = [1, 0, 0]
    elif (row['weather'] == 'sunrise'):
        label = [0, 1, 0]
    elif (row['weather'] == 'cloudy'):
        label = [0, 0, 1]
    elif (row['weather'] == 'shine'):
        label = [0, 0, 0]
    Y_test_data.append(label)

Y_test_data = np.array(Y_test_data)
#Y_test_data.shape

# Test2
X_test_data = []
for i in tqdm(range(test_data.shape[0])):
    path = 'metadata-test/test/' + str(test_data['file_name'][i])
    klasa = test_data['weather'][i]
    img = image.load_img(path, target_size=(img_width, img_height, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    X_test_data.append(img)
    img = img.reshape(1, img_width, img_height, 3)  # 3 ->rgb
    y_prob = model.predict(img)

X_test_data = np.array(X_test_data)
#X_test_data.shape
X_train, X_test, y_train, y_test = train_test_split(X_test_data, Y_test_data, random_state=0, test_size=0.15)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)  # tensorflow
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)
