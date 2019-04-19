import numpy as np 
import os 
import cv2 as cv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv3D, Dense, Dropout, Flatten

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
    datasets = []
    labels_OneHot = []
    for path in data_path:
        filename = 'data/' + path[0] + '/' + path[1]
        cap = cv.VideoCapture(filename)        
        datasets.append(read_video_data(cap))
        labels_OneHot.append(labels_classes.index(path[0]))
    datasets = np.array(datasets)
    datasets = datasets.reshape([datasets.shape[0], 30, 240, 320, 1])
    labels_OneHot = np.array(labels_OneHot)
    labels_OneHot = np_utils.to_categorical(labels_OneHot)
    
    return datasets, labels_OneHot

data_path = load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

labels_classes = []
for e in labels:
    if e not in labels_classes:
        labels_classes.append(e)

datasets, all_labels = load_data(data_path, labels, labels_classes)

# 建立模型
model = Sequential()
model.add(Conv3D(filters=16, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same', input_shape=(30, 240, 320, 1), activation='relu'))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=21, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x = datasets, y=all_labels, validation_split=0.3, epochs=10, batch_size=20)