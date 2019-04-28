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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder('float32', shape=[None, 30, 240, 320, 1])

with tf.name_scope('C1_Conv'):
    W1 = weight([5, 5, 1, 4])
    b1 = bias([4])
    x = x.reshape([-1, 240, 320, 1])
    Conv1 = conv2d(x, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

with tf.name_scope('C2_Conv'):
    W2 = weight([5, 5, 4, 8])
    b2 = bias([8])
    Conv2 = conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)

with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 38400])

with tf.name_scope('Hidden_Layer1'):
    W3 = weight([38400, 1000])
    b3 = bias([1000])
    D_Hidden1 = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)

with tf.name_scope('Hidden_Layer2'):
    W4 = weight([1000, 100])
    b4 = bias([100])
    D_Hidden2 = tf.nn.relu(tf.matmul(D_Hidden1, W4) + b4)

with tf.name_scope('Hidden_Layer3'):
    W5 = weight([100, 1])
    b5 = bias([1])
    D_Hidden3 = tf.nn.relu(tf.matmul(D_Hidden2, W5) + b5)

with tf.name_scope('Output_Layer'):
    V_Hidden = D_Hidden3.reshape([-1, 30])
    W6 = weight([30, 21])
    b6 = bias([21])
    y_predict = tf.nn.softmax(tf.matmul(V_Hidden, W6) + b6)

with tf.name_scope('optimizer'):
    y = tf.placeholder('float32', shape=[None, 21])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    optimizer = tf.train.AdamOptimizer(1e-8).minimize(loss)

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

test_data1, test_label1 = load_data.load_data(data_path[index[696: 746], :], labels, labels_classes)
test_data1 = test_data1.reshape([-1, 30, 240, 320, 1])
test_data1 = test_data1 / 255.

test_data2, test_label2 = load_data.load_data(data_path[index[746: 796], :], labels, labels_classes)
test_data2 = test_data2.reshape([-1, 30, 240, 320, 1])
test_data2 = test_data2 / 255.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for i in range(29):
            train_data, train_label = load_data.load_data(data_path[index[i * batch_size: (i + 1) * batch_size], :], labels, labels_classes)
            train_data = train_data.reshape([24, 30, 240, 320, 1])
            train_data = train_data / 255.
            
            sess.run(optimizer, feed_dict = {x: train_data, y: train_label})
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict = {x: train_data, y: train_label})
            print('Epoch ', epoch + 1, 'Batch', i, 'train_acc = ', train_acc, ', train_loss = ', train_loss)
        loss, acc = sess.run([loss, accuracy], feed_dict = {x: test_data1, y: test_label1})
        print('Epoch ', epoch + 1, 'acc1 = ', acc, ', loss1 = ', loss)
        loss, acc = sess.run([loss, accuracy], feed_dict = {x: test_data2, y: test_label2})
        print('Epoch ', epoch + 1, 'acc2 = ', acc, ', loss2 = ', loss)
        # test_loss1, test_acc1 = sess.run([loss_function, accuracy], feed_dict = {x: test_data1, y: test_label1})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss1, 'test1_acc = ', test_acc1)
        # test_loss2, test_acc2 = sess.run([loss_function, accuracy], feed_dict = {x: test_data2, y: test_label2})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss2, 'test1_acc = ', test_acc2)