import numpy as np 
import os 
import cv2 as cv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv3D, Dense, Dropout, Flatten

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

all_labels = range(21)

def load_data(data_path, all_labels):
    datasets = []
    i = 0
    for path in data_path:
        filename = 'data/' + path[0] + '/' + path[1]
        cap = cv.VideoCapture(filename)        
        datasets.append(read_video_data(cap))
        if i == 50:
            break
        i = i + 1
    datasets = np.array(datasets)
    
    return datasets

data_path = load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

datasets = load_data(data_path, all_labels)
print(np.shape(datasets))
print(np.shape(labels))