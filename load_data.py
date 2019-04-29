import numpy as np 
import tensorflow as tf 
import os 
import cv2 as cv
from keras.utils import np_utils
import csv

# 预处理
def load_data_path():
    path = 'data'
    files = os.listdir(path)
    s = []
    for _file in files:
        data_files = os.listdir(path + '/' + _file)
        for data_file in data_files:
            s.append([_file, data_file])
    return s

def read_video_data(cap):
    video_data = []
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            video_data.append(frame)
        else:
            video_data.append(np.zeros(shape=(240, 320, 3)))
    return video_data

def load_data(data_path, labels, labels_classes):
    datasets = []
    labels_OneHot = []
    i = 0
    for path in data_path:
        filename = 'data/' + path[0] + '/' + path[1]
        # print('load data from: ', filename, i)
        i = i + 1
        cap = cv.VideoCapture(filename)        
        datasets.append(read_video_data(cap))
        labels_OneHot.append(labels_classes.index(path[0]))
    # datasets = datasets.reshape([datasets.shape[0], 30, 240, 320, 1])
    datasets = np.array(datasets)
    labels_OneHot = np.array(labels_OneHot)
    labels_OneHot = np_utils.to_categorical(labels_OneHot, num_classes=21)
    
    return datasets, labels_OneHot

data_path = load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

labels_classes = []
for e in labels:
    if e not in labels_classes:
        labels_classes.append(e)

index = np.arange(928)
# np.random.shuffle(index)

train_data, train_label = load_data(data_path[index[:60]], labels, labels_classes)
print(train_data.shape)
# train_data = train_data.reshape([-1, 30, 240, 320, 3])

# print(np.shape(train_data[0][0]))
train_data = train_data / 255.


for i in range(60):
    for j in range(30):
        # print(train_data[i][j])
        # frame = cv.UMat(train_data[i][j])
        cv.imshow('' + labels[index[i]], train_data[i][j])
        cv.waitKey(10)
    cv.destroyAllWindows()

    
# load_data(data_path, labels, labels_classes)

# datasets, all_labels = load_data(data_path, labels, labels_classes)

# # 建立模型
# model = Sequential()
# model.add(Conv3D(filters=2, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same', input_shape=(30, 240, 320, 1), activation='relu', data_format='channels_last'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', dim_ordering='tf'))
# model.add(Dropout(0.3))

# model.add(Conv3D(filters=4, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same', data_format='channels_last'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', dim_ordering='tf'))
# model.add(Dropout(0.3))

# model.add(Conv3D(filters=8, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same', data_format='channels_last'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', dim_ordering='tf'))
# model.add(Dropout(0.3))

# model.add(Flatten())

# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=21, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())

# train_history = model.fit(x = datasets, y=all_labels, validation_split=0.3, epochs=10, batch_size=20)
