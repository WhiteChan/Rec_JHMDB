import numpy as np 
import tensorflow as tf 
import os 
import cv2 as cv
from keras.utils import np_utils

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
    datasets = np.array([], dtype=np.float)
    labels_OneHot = np.array([], dtype=np.float)
    for path in data_path:
        filename = 'data/' + path[0] + '/' + path[1]
        cap = cv.VideoCapture(filename)        
        datasets = np.append(datasets, read_video_data(cap))
        labels_OneHot = np.append(labels_OneHot, labels_classes.index(path[0]))
    datasets = datasets.reshape([datasets.shape[0], 30, 240, 320, 1])
    labels_OneHot = np_utils.to_categorical(labels_OneHot)
    
    return datasets, labels_OneHot

data_path = load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

labels_classes = []
for e in labels:
    if e not in labels_classes:
        labels_classes.append(e)

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

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 30, 240, 320, 1])

with tf.name_scope('C1_Conv'):
    W1 = weight([3, 3, 3, 1, 2])
    b1 = bias([2])
    Conv1 = conv3d(x, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_3x3(C1_Conv)

with tf.name_scope('C2_Conv'):
    W2 = weight([3, 3, 3, 2, 4])
    b2 = bias([4])
    Conv2 = conv3d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)

with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_3x3(C2_Conv)

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C1_Pool, [-1, 134400])

with tf.name_scope('Hidden_Layer'):
    W3 = weight([134400, 100])
    b3 = bias([100])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)

with tf.name_scope('Output_Layer'):
    W4 = weight([100, 21])
    b4 = bias([21])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden, W4) + b4)

with tf.name_scope('optimizer'):
    y = tf.placeholder("float", shape=[None, 21])
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_function)

with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

batch_size = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        train_x, train_y = load_data(data_path, labels, labels_classes)
        for i in range(18):
            loss, acc = sess.run([loss_function, accuracy], feed_dict = {x: train_x[i * batch_size: i * batch_size + batch_size], y: train_y[i * batch_size: i * batch_size + batch_size]})
            print('batch ', i, 'acc = ', acc, ', loss = ', loss)