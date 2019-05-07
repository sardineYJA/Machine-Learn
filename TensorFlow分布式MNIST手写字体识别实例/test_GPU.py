import tensorflow as tf

# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
    c = a + b

# allow_soft_placement=True 表示使用不能使用显卡时使用cpu
# log_device_placement=False 不打印日志，不然会刷屏
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
