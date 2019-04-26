import numpy as np 
import tensorflow as tf 
import load_data

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

with tf.name_scope('C3_Conv'):
    W3 = weight([3, 3, 3, 4, 8])
    b3 = bias([8])
    Conv3 = conv3d(C2_Pool, W3) + b3
    C3_Conv = tf.nn.relu(Conv3)

with tf.name_scope('C3_Pool'):
    C3_Pool = max_pool_3x3(C3_Conv)

with tf.name_scope('C4_Conv'):
    W4 = weight([3, 3, 3, 8, 16])
    b4 = bias([16])
    Conv4 = conv3d(C3_Pool, W4) + b4
    C4_Conv = tf.nn.relu(Conv4)

with tf.name_scope('C4_Pool'):
    C4_Pool = max_pool_3x3(C4_Conv)

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C4_Pool, [-1, 9600])

with tf.name_scope('Hidden_Layer1'):
    W5 = weight([9600, 3000])
    b5 = bias([3000])
    D_Hidden1 = tf.nn.relu(tf.matmul(D_Flat, W5) + b5)

with tf.name_scope('Hidden_Layer2'):
    W6 = weight([3000, 1000])
    b6 = bias([1000])
    D_Hidden2 = tf.nn.relu(tf.matmul(D_Hidden1, W6) + b6)

with tf.name_scope('Hidden_Layer3'):
    W7 = weight([1000, 100])
    b7 = bias([100])
    D_Hidden3 = tf.nn.relu(tf.matmul(D_Hidden2, W7) + b7)

with tf.name_scope('Output_Layer'):
    W8 = weight([100, 21])
    b8 = bias([21])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden3, W8) + b8)

with tf.name_scope('optimizer'):
    y = tf.placeholder("float", shape=[None, 21])
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_function)

with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

data_path = load_data.load_data_path()
data_path = np.array(data_path)
labels = data_path[:, 0]

labels_classes = []
for e in labels:
    if e not in labels_classes:
        labels_classes.append(e)

batch_size = 24
index = np.arange(928)
np.random.shuffle(index)
# train_data, train_label = load_data.load_data(data_path[index[0: batch_size], :], labels, labels_classes)
# train_data = train_data.reshape([24, 30, 240, 320, 1])
# print(np.shape(train_data))
# print(np.shape(train_label))
# train_label = labels[index[0: 696], :]
# test_data = datas[index[696:], :, :, :, :]
# test_label = labels[index[696: ], :]



# test_data1, test_label1 = load_data.load_data(data_path[index[696: 796], :], labels, labels_classes)
# test_data1 = test_data1.reshape([-1, 30, 240, 320, 1])
# test_data1 = test_data1 / 255.

# test_data2, test_label2 = load_data.load_data(data_path[index[796: ], :], labels, labels_classes)
# test_data2 = test_data2.reshape([-1, 30, 240, 320, 1])
# test_data2 = test_data2 / 255.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for i in range(29):
            train_data, train_label = load_data.load_data(data_path[index[i * batch_size: (i + 1) * batch_size], :], labels, labels_classes)
            train_data = train_data.reshape([24, 30, 240, 320, 1])
            train_data = train_data / 255.
            
            sess.run(optimizer, feed_dict = {x: train_data, y: train_label})
            loss, acc = sess.run([loss_function, accuracy], feed_dict = {x: train_data, y: train_label})
            print('Epoch ', epoch + 1, 'batch ', i, 'acc = ', acc, ', loss = ', loss)
        # test_loss1, test_acc1 = sess.run([loss_function, accuracy], feed_dict = {x: test_data1, y: test_label1})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss1, 'test1_acc = ', test_acc1)
        # test_loss2, test_acc2 = sess.run([loss_function, accuracy], feed_dict = {x: test_data2, y: test_label2})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss2, 'test1_acc = ', test_acc2)