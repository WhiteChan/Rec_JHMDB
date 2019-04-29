import pandas as pd 
import numpy as np
import cv2 as cv 
import os 
from keras.utils import np_utils

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
#     i = 0
#     while(i < 100):
#         while(True):
#             ret, frame = cap.read()
#             if ret and i < 100:
#                 video_data.append(frame)
#                 cv.imshow('data', frame)
#                 cv.waitKey(100)
#                 i = i + 1
#             else:
#                 break
#         if i < 100:
#             video_data.append(np.zeros(shape=(240, 320, 3)))
#             i = i + 1
    for i in range(100):
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
        print('load data from: ', filename, i)
        i = i + 1
        cap = cv.VideoCapture(filename)        
        datasets.append(read_video_data(cap))
        labels_OneHot.append(labels_classes.index(path[0]))
    # datasets = datasets.reshape([datasets.shape[0], 30, 240, 320, 1])
    # for i in range(100):
    #     cv.imshow('datasets', np.array(datasets[0][i]))
    #     cv.waitKey(100)
    # datasets = np.array(datasets)  # ******************************
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
np.random.shuffle(index)

train_data, train_label = load_data(data_path[index[:1]], labels, labels_classes)
# print(train_data.shape)
# train_data = train_data.reshape([-1, 30, 240, 320, 3])

print(np.shape(train_data))
# print(train_data[0])

train_data = np.array(train_data) / 255.

for i in range(100):
    cv.imshow('test', np.array(train_data[0][i]))
    cv.waitKey(100)

# for i in range(30):
#     for j in range(30):
#         # print(train_data[i][j])
#         # frame = cv.UMat(train_data[i][j])
#         cv.imshow('' + labels[index[i]], train_data[i][j])
#         cv.waitKey(10)
#     cv.destroyAllWindows()
