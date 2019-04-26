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
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            video_data.append(frame)
        else:
            video_data.append(np.zeros(shape=(240, 320)))
    return video_data

def load_data(data_path, labels, labels_classes):
    video_labels = []
    i = 0
    data_out = open('data.csv', 'a', newline='')
    label_out = open('label.csv', 'a', newline='')
    data_csv_write = csv.writer(data_out, dialect='excel')
    label_csv_write = csv.writer(label_out, dialect='excel')
    for path in data_path:
        filename = 'data/' + path[0] + '/' + path[1]
        print('load data from: ', filename, i)
        i = i + 1
        cap = cv.VideoCapture(filename)
        img_data = read_video_data(cap)
        img_data = np.reshape(img_data, newshape=(30, 240 * 320))
        for k in range(30):
            data_csv_write.writerow(img_data[k])
        video_labels.append(labels_classes.index(path[0]))
    label_csv_write.writerow(video_labels)

data_path = load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

labels_classes = []
for e in labels:
    if e not in labels_classes:
        labels_classes.append(e)
    
load_data(data_path, labels, labels_classes)

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

# # train_history = model.fit(x = datasets, y=all_labels, validation_split=0.3, epochs=10, batch_size=20)
