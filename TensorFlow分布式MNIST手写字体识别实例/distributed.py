# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/mnist',
                    'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100,
                     'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000,
                     'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')

# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '192.168.25.101:22221',
                    'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '192.168.25.102:22221,192.168.25.103:22221,192.168.25.104:22221',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS


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
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2'])+_biases['bd2'], name='fc2')
    dense3 = tf.nn.relu(tf.matmul(dense2, _weights['wd3'])+_biases['bd3'], name='fc3')
    dense2 = tf.nn.dropout(dense3, _dropout)
    out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
    return out


def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    print('ps_spec : ', ps_spec)          # ['192.168.25.101:22221']
    print('worker_spec : ', worker_spec)  # ['192.168.25.102:22221', '192.168.25.103:22221']

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(
       cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 全局训练步数
        x = tf.placeholder(tf.float32, [None, 28*28])
        y = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        # 最小的卷积核3x3，学习细节特征，标准差设置小一点，loss才正常，收敛很快，否则精确度上不去
        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)),
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
            'wd1': tf.Variable(tf.random_normal([4*4*128, 1024], stddev=0.01)),
            'wd2': tf.Variable(tf.random_normal([1024, 1024], stddev=0.01)),
            'wd3': tf.Variable(tf.random_normal([1024, 512], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([512, 10], stddev=0.01))}

        biases = {
            'bc1': tf.Variable(tf.random_normal([32], stddev=0.01)),
            'bc2': tf.Variable(tf.random_normal([64], stddev=0.01)),
            'bc3': tf.Variable(tf.random_normal([128], stddev=0.01)),
            'bd1': tf.Variable(tf.random_normal([1024], stddev=0.01)),
            'bd2': tf.Variable(tf.random_normal([1024], stddev=0.01)),
            'bd3': tf.Variable(tf.random_normal([512], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([10], stddev=0.01))}

        # 三层网络，交叉熵，优化器
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, global_step=global_step)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        # 创建临时目录
        train_dir = tempfile.mkdtemp()
        # 创建Supervisor对象，将要保存checkpoints以及summaries的目录路径传递给该对象
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:  # task==0
            print('Worker %d: 初始化 session...' % FLAGS.task_index)
        else:         # task!=0
            print('Worker %d: 等待初始化 session...' % FLAGS.task_index)

        # managed_session在异步模式就是参数初始化完成之后，大家就可以开始干活了。 
        # prepare_or_wait_for_session在同步模式，不但参数初始化完成，还得主节点也准备好了，其他节点才开始干活。
        sess = sv.prepare_or_wait_for_session(server.target)
        print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y: batch_ys, keep_prob: 1.}
            _, step = sess.run([optimizer, global_step], feed_dict=train_feed)
            local_step += 1

            if local_step % 10 == 0:  # 每训练100次输出结果
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print('Worker %d: traing step %d dome (global step:%d)' % (FLAGS.task_index, local_step, step))
                print("Loss= " + "{:.6f}".format(loss) + ",Accuracy= " + "{:.5f}".format(acc))

            if step >= FLAGS.train_steps:  # 达到训练次数要求，退出
                break

        time_end = time.time()
        print('Training ends @ %f' % time_end)
        print('Training elapsed time:%f s' % (time_end - time_begin))

        # 验证集
        val_feed = {x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.}
        print("Validation cost:", sess.run(cost, feed_dict=val_feed))
        print("Validation Accuracy:", sess.run(accuracy, feed_dict=val_feed))
        # 测试集
        test_feed = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.}
        print("Testing cost:", sess.run(cost, feed_dict=test_feed))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict=test_feed))
    sess.close()


if __name__ == '__main__':
    tf.app.run()
