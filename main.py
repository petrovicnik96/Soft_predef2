import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
from tqdm import tqdm

csvData = pd.read_csv('metadata/train/train_labels.csv')

img_width = 350
img_height = 350

X = []
Y = []

for i in tqdm(range(csvData.shape[1])):
    path = 'metadata/train/' + csvData['file_name'][i]
    img = image.load_img(path, target_size=(img_width, img_height, 2))
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

X = np.array(X)
# print(len(X))
y = csvData.drop(columns=['file_name', 'weather'])[0:csvData.shape[0]]
y.head()
