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

x = tf.placeholder('float32', shape=[None, 30, 240, 320, 3])

with tf.name_scope('C1_Conv'):
    W1 = weight([5, 5, 3, 64])
    b1 = bias([64])
    x_reshape = tf.reshape(x, [-1, 240, 320, 3])
    Conv1 = conv2d(x_reshape, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

with tf.name_scope('C2_Conv'):
    W2 = weight([5, 5, 64, 128])
    b2 = bias([128])
    Conv2 = conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)

with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)

with tf.name_scope('C3_Conv'):
    W3 = weight([5, 5, 128, 256])
    b3 = bias([256])
    Conv3 = conv2d(C2_Pool, W3) + b3
    C3_Conv = tf.nn.relu(Conv3)

with tf.name_scope('C3_Pool'):
    C3_Pool = max_pool_2x2(C3_Conv)

# ***************************
with tf.name_scope('C4_Conv'):
    W31 = weight([5, 5, 256, 512])
    b31 = bias([512])
    Conv4 = conv2d(C3_Pool, W31) + b31
    C4_Conv = tf.nn.relu(Conv4)

with tf.name_scope('C4_Pool'):
    C4_Pool = max_pool_2x2(C4_Conv)

with tf.name_scope('C5_Conv'):
    W32 = weight([5, 5, 512, 1024])
    b32 = bias([1024])
    Conv5 = conv2d(C4_Pool, W32) + b32
    C5_Conv = tf.nn.relu(Conv5)

with tf.name_scope('C5_Pool'):
    C5_Pool = max_pool_2x2(C5_Conv)

with tf.name_scope('C6_Conv'):
    W33 = weight([5, 5, 1024, 2048])
    b33 = bias([2048])
    Conv6 = conv2d(C5_Pool, W33) + b33
    C6_Conv = tf.nn.relu(Conv6)

with tf.name_scope('C6_Pool'):
    C6_Pool = max_pool_2x2(C6_Conv)
# ***************************

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C6_Pool, [-1, 40960])

with tf.name_scope('Hidden_Layer1'):
    W4 = weight([40960, 1000])
    b4 = bias([1000])
    D_Hidden1 = tf.nn.relu(tf.matmul(D_Flat, W4) + b4)

with tf.name_scope('Hidden_Layer2'):
    W5 = weight([1000, 100])
    b5 = bias([100])
    D_Hidden2 = tf.nn.relu(tf.matmul(D_Hidden1, W5) + b5)

with tf.name_scope('Hidden_Layer3'):
    W6 = weight([100, 21])
    b6 = bias([21])
    D_Hidden3 = tf.nn.relu(tf.matmul(D_Hidden2, W6) + b6)

with tf.name_scope('Output_Layer'):
    V_Hidden = tf.reshape(D_Hidden3, [-1, 21 * 30])
    W7 = weight([21 * 30, 21])
    b7 = bias([21])
    y_predict = tf.nn.softmax(tf.matmul(V_Hidden, W7) + b7)

with tf.name_scope('optimizer'):
    y = tf.placeholder('float32', shape=[None, 21])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

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

batch_size = 6
index = np.arange(928)
np.random.shuffle(index)

# test_data1, test_label1 = load_data.load_data(data_path[index[696: 746], :], labels, labels_classes)
# test_data1 = test_data1.reshape([-1, 30, 240, 320, 1])
# test_data1 = test_data1 / 255.

# test_data2, test_label2 = load_data.load_data(data_path[index[746: 796], :], labels, labels_classes)
# test_data2 = test_data2.reshape([-1, 30, 240, 320, 1])
# test_data2 = test_data2 / 255.

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN', sess.graph)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(1000):
#         for i in range(116):
#             train_data, train_label = load_data.load_data(data_path[index[i * batch_size: (i + 1) * batch_size], :], labels, labels_classes)
#             train_data = train_data / 255.

#             # logist_ = sess.run(C3_Pool, feed_dict={x: train_data})
#             # print(np.shape(logist_), i)
#             # break

#             sess.run(optimizer, feed_dict = {x: train_data, y: train_label})
#             train_loss, train_acc = sess.run([loss, accuracy], feed_dict = {x: train_data, y: train_label})
#             print('Epoch ', epoch + 1, 'Batch', i, 'train_acc = ', train_acc, ', train_loss = ', train_loss)
        # loss, acc = sess.run([loss, accuracy], feed_dict = {x: test_data1, y: test_label1})
        # print('Epoch ', epoch + 1, 'acc1 = ', acc, ', loss1 = ', loss)
        # loss, acc = sess.run([loss, accuracy], feed_dict = {x: test_data2, y: test_label2})
        # print('Epoch ', epoch + 1, 'acc2 = ', acc, ', loss2 = ', loss)
        # test_loss1, test_acc1 = sess.run([loss_function, accuracy], feed_dict = {x: test_data1, y: test_label1})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss1, 'test1_acc = ', test_acc1)
        # test_loss2, test_acc2 = sess.run([loss_function, accuracy], feed_dict = {x: test_data2, y: test_label2})
        # print('Epoch ', epoch + 1, ': test1_loss = ', test_loss2, 'test1_acc = ', test_acc2)