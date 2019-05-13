import numpy as np 
import os 
import cv2 as cv
from keras.utils import np_utils
import random
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 

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
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            video_data.append(frame)
    return video_data[int(random.random() * 10)]

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

batch_size = 12
index = np.arange(928)
np.random.shuffle(index)

train_data, train_label = load_data(data_path[index], labels, labels_classes)

# build model
# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(240, 320, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(units=1024, activation='relu'))
# model.add(Dense(units=512, activation='relu'))
# model.add(Dense(units=21, activation='softmax'))

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)

predictions = Dense(21, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap = 'binary')
        title = "label = " + str(labels_classes[np.argmax(labels[idx])])
        if len(prediction) > 0:
            title += ", prediction = " + str(prediction[idx])

        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

# plot_images_labels_prediction(train_data, train_label, '', idx=0, num=10)

early_stopper = EarlyStopping(patience=10)

train_history = model.fit(x = train_data[:696], y=train_label[:696], validation_split=0.2, epochs=1000, batch_size=20, callbacks=[early_stopper])

scores = model.evaluate(train_data[696:], train_label[696:])
print('loss = ', scores[0], ', acc = ', scores[1])