import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 卷积层
def conv2d(image, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
        image, w, strides=[1, 1, 1, 1], padding='SAME'), b))


# 池化层，k步长
def max_pooling(image, k):
    return tf.nn.max_pool(
        image, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# 卷积网络
def conv_net(_X, _weights, _biases, _dropout):
    # 三层网络
    _X = tf.reshape(_X, [-1, 28, 28, 1])
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    conv1 = max_pooling(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)

    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    conv2 = max_pooling(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, keep_prob=_dropout)

    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    conv3 = max_pooling(conv3, k=2)
    conv3 = tf.nn.dropout(conv3, keep_prob=_dropout)

    dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(
        tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense2 = tf.nn.relu(
        tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
    dense3 = tf.nn.relu(
        tf.matmul(dense2, weights['wd3']) + _biases['bd3'], name='fc3')
    dense2 = tf.nn.dropout(dense3, _dropout)
    out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
    return out


if __name__ == '__main__':
    # 数据下载并读取
    mnist = input_data.read_data_sets("mnist/", one_hot=True)

    # 参数
    learning_rate = 0.001     # 学习率
    training_iters = 100000    # 训练迭代次数
    batch_size = 100          # batch大小
    display_step = 10         # 打印步长
    n_input = 784             # 输入特征
    n_classes = 10            # 10个分类
    dropout = 0.9             # 全连接多设置大一些

    with tf.device('/gpu:0'):
        # 输入输出占位符
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)

        # 最小的卷积核3x3，学习细节特征，标准差设置小一点，loss才正常，收敛很快，否则精确度上不去
        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)),
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
            'wd1': tf.Variable(tf.random_normal([4 * 4 * 128, 1024], stddev=0.01)),
            'wd2': tf.Variable(tf.random_normal([1024, 1024], stddev=0.01)),
            'wd3': tf.Variable(tf.random_normal([1024, 512], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([512, n_classes], stddev=0.01))}

        biases = {
            'bc1': tf.Variable(tf.random_normal([32], stddev=0.01)),
            'bc2': tf.Variable(tf.random_normal([64], stddev=0.01)),
            'bc3': tf.Variable(tf.random_normal([128], stddev=0.01)),
            'bd1': tf.Variable(tf.random_normal([1024], stddev=0.01)),
            'bd2': tf.Variable(tf.random_normal([1024], stddev=0.01)),
            'bd3': tf.Variable(tf.random_normal([512], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))}

        # 三层网络，交叉熵，优化器
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)           # 初始化变量
        step = 1
        print(step)
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                     x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={
                               x: batch_xs, y: batch_ys, keep_prob: 1.})
                loss = sess.run(cost, feed_dict={
                                x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        print("Optimization Finished!")
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.}))
