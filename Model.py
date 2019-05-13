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

x = tf.placeholder("float32", shape=[None, 30, 240, 320, 3])

with tf.name_scope('C1_Conv'):
    W1 = weight([3, 3, 3, 3, 64])
    b1 = bias([64])
    Conv1 = conv3d(x, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope('C1_Pool'):
    C1_Pool = tf.nn.max_pool3d(C1_Conv, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')

with tf.name_scope('C2_Conv'):
    W2 = weight([3, 3, 3, 64, 128])
    b2 = bias([128])
    Conv2 = conv3d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)

with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_3x3(C2_Conv)

with tf.name_scope('C3a_Conv'):
    W3a = weight([3, 3, 3, 128, 256])
    b3a = bias([256])
    Conv3a = conv3d(C2_Pool, W3a) + b3a
    C3a_Conv = tf.nn.relu(Conv3a)

with tf.name_scope('C3b_Conv'):
    W3b = weight([3, 3, 3, 256, 256])
    b3b = bias([256])
    Conv3b = conv3d(C3a_Conv, W3b) + b3b
    C3b_Conv = tf.nn.relu(Conv3b)

with tf.name_scope('C3_Pool'):
    C3_Pool = max_pool_3x3(C3b_Conv)

with tf.name_scope('C4a_Conv'):
    W4a = weight([3, 3, 3, 256, 512])
    b4a = bias([512])
    Conv4a = conv3d(C3_Pool, W4a) + b4a
    C4a_Conv = tf.nn.relu(Conv4a)

with tf.name_scope('C4b_Conv'):
    W4b = weight([3, 3, 3, 512, 512])
    b4b = bias([512])
    Conv4b = conv3d(C4a_Conv, W4b) + b4b
    C4b_Conv = tf.nn.relu(Conv4b)

with tf.name_scope('C4_Pool'):
    C4_Pool = max_pool_3x3(C4b_Conv)

with tf.name_scope('C5a_Conv'):
    W5a = weight([3, 3, 3, 512, 512])
    b5a = bias([512])
    Conv5a = conv3d(C4_Pool, W5a) + b5a
    C5a_Conv = tf.nn.relu(Conv5a)

with tf.name_scope('C5b_Conv'):
    W5b = weight([3, 3, 3, 512, 512])
    b5b = bias([512])
    Conv5b = conv3d(C5a_Conv, W5b) + b5b
    C5b_Conv = tf.nn.relu(Conv5b)

with tf.name_scope('C5_Pool'):
    C5_Pool = max_pool_3x3(C5b_Conv)

with tf.name_scope('C6a_Conv'):
    W6a = weight([3, 3, 3, 512, 1024])
    b6a = bias([1024])
    Conv6a = conv3d(C5_Pool, W6a) + b6a
    C6a_Conv = tf.nn.relu(Conv6a)

with tf.name_scope('C6b_Conv'):
    W6b = weight([3, 3, 3, 1024, 1024])
    b6b = bias([1024])
    Conv6b = conv3d(C6a_Conv, W6b) + b6b
    C6b_Conv = tf.nn.relu(Conv6b)

with tf.name_scope('C6_Pool'):
    C6_Pool = max_pool_3x3(C6b_Conv)

with tf.name_scope('C7_Conv'):
    W7 = weight([3, 3, 3, 1024, 1024])
    b7 = bias([1024])
    Conv7 = conv3d(C6_Pool, W7) + b7
    C7_Conv = tf.nn.relu(Conv7)

with tf.name_scope('C7_Pool'):
    C7_Pool = max_pool_3x3(C7_Conv)

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C7_Pool, [-1, 6144])

with tf.name_scope('Hidden_Layer1'):
    W5 = weight([6144, 100])
    b5 = bias([100])
    D_Hidden1 = tf.nn.relu(tf.matmul(D_Flat, W5) + b5)

# with tf.name_scope('Hidden_Layer2'):
#     W6 = weight([3000, 1000])
#     b6 = bias([1000])
#     D_Hidden2 = tf.nn.relu(tf.matmul(D_Hidden1, W6) + b6)

# with tf.name_scope('Hidden_Layer3'):
#     W7 = weight([1000, 100])
#     b7 = bias([100])
#     D_Hidden3 = tf.nn.relu(tf.matmul(D_Hidden2, W7) + b7)

with tf.name_scope('Output_Layer'):
    W8 = weight([100, 21])
    b8 = bias([21])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden1, W8) + b8)

with tf.name_scope('optimizer'):
    y = tf.placeholder("float32", shape=[None, 21])
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

batch_size = 12
index = np.arange(928)
np.random.shuffle(index)

# test_data1, test_label1 = load_data.load_data(data_path[index[696: 746], :], labels, labels_classes)
# test_data1 = test_data1.reshape([-1, 30, 240, 320, 1])
# test_data1 = test_data1 / 255.

# test_data2, test_label2 = load_data.load_data(data_path[index[746: 796], :], labels, labels_classes)
# test_data2 = test_data2.reshape([-1, 30, 240, 320, 1])
# test_data2 = test_data2 / 255.


# sess = tf.Session()

# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('log/CNN', sess.graph)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for i in range(696 / batch_size):
            train_data, train_label = load_data.load_data(data_path[index[i * batch_size: (i + 1) * batch_size], :], labels, labels_classes)
            train_data = train_data / 255.

            # logist_ = sess.run(C3_Pool, feed_dict={x: train_data})
            # print(np.shape(logist_), i)
            # break

            sess.run(optimizer, feed_dict = {x: train_data, y: train_label})
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict = {x: train_data, y: train_label})
            print('Epoch ', epoch + 1, 'Batch', i, 'train_acc = ', train_acc, ', train_loss = ', train_loss)

# batch_size = 24
# index = np.arange(928)
# np.random.shuffle(index)
# # train_data, train_label = load_data.load_data(data_path[index[0: batch_size], :], labels, labels_classes)
# # train_data = train_data.reshape([24, 30, 240, 320, 1])
# # print(np.shape(train_data))
# # print(np.shape(train_label))
# # train_label = labels[index[0: 696], :]
# # test_data = datas[index[696:], :, :, :, :]
# # test_label = labels[index[696: ], :]



# test_data1, test_label1 = load_data.load_data(data_path[index[696: 746], :], labels, labels_classes)
# test_data1 = test_data1.reshape([-1, 30, 240, 320, 1])
# test_data1 = test_data1 / 255.

# test_data2, test_label2 = load_data.load_data(data_path[index[746: 796], :], labels, labels_classes)
# test_data2 = test_data2.reshape([-1, 30, 240, 320, 1])
# test_data2 = test_data2 / 255.

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(1000):
#         for i in range(29):
#             train_data, train_label = load_data.load_data(data_path[index[i * batch_size: (i + 1) * batch_size], :], labels, labels_classes)
#             train_data = train_data.reshape([24, 30, 240, 320, 1])
#             train_data = train_data / 255.
            
#             sess.run(optimizer, feed_dict = {x: train_data, y: train_label})
#             train_loss, train_acc = sess.run([loss_function, accuracy], feed_dict = {x: train_data, y: train_label})
#             print('Epoch ', epoch + 1, 'Batch', i, 'train_acc = ', train_acc, ', train_loss = ', train_loss)
#         loss, acc = sess.run([loss_function, accuracy], feed_dict = {x: test_data1, y: test_label1})
#         print('Epoch ', epoch + 1, 'acc1 = ', acc, ', loss1 = ', loss)
#         loss, acc = sess.run([loss_function, accuracy], feed_dict = {x: test_data2, y: test_label2})
#         print('Epoch ', epoch + 1, 'acc2 = ', acc, ', loss2 = ', loss)
#         # test_loss1, test_acc1 = sess.run([loss_function, accuracy], feed_dict = {x: test_data1, y: test_label1})
#         # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss1, 'test1_acc = ', test_acc1)
#         # test_loss2, test_acc2 = sess.run([loss_function, accuracy], feed_dict = {x: test_data2, y: test_label2})
#         # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss2, 'test1_acc = ', test_acc2)