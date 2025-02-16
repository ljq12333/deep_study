import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
"""
通过调节参数来提高准确率
1）学习率
2）随机初始化的权重，偏置的值
3）选择好的优化器
4）调整网络结构
"""


def create_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape))


def create_model(x):
    """
    构建卷积神经网络模型
    :param x:
    :return:
    """
    y_predict = 0
    with tf.variable_scope("conv1"):
        #卷积层
        #将X的shape进行修改，改为四阶形状
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv_w = create_weights(shape=[5, 5, 1, 32])
        conv_basic = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=input_x, filter=conv_w, strides=[1, 1, 1, 1], padding="SAME") + conv_basic
        #激活层
        relu1_x = tf.nn.relu(conv1_x)
        #池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    #第二卷积神经网络层
    with tf.variable_scope("conv2"):
        conv_w = create_weights(shape=[5, 5, 32, 64])
        conv_basic = create_weights(shape=[64])
        conv1_x = tf.nn.conv2d(input=pool1_x, filter=conv_w, strides=[1, 1, 1, 1], padding="SAME") + conv_basic
        # 激活层
        relu2_x = tf.nn.relu(conv1_x)
        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    #全连接层
    with tf.variable_scope("lj"):
        lj_x = tf.reshape(pool2_x, shape=[-1, 7*7*64],)
        w = tf.Variable(tf.random_normal(shape=[7*7*64, 10]), name='w')
        b = tf.Variable(tf.random_normal(shape=[10]), name='b')
        y_predict = tf.matmul(lj_x, w) + b
    return y_predict


def full_collection():
    """
    用全连接来对手写的数字进行识别
    :return:
    """
    #准备数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    #构建模型
    y_predict = create_model(x)
    #构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict, name='y'))
    #优化损失
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
    #正确率计算
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    avg_ = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        num_batch = mnist.train.num_examples
        for i in range(10):
            for _ in range(int(num_batch/100)):
                image, label = mnist.train.next_batch(100)
                _, loss, avg_new = sess.run([optimizer, error, avg_], feed_dict={x: image, y_true: label})
            print("第%d次优化损失值为---%s--正确率为--%f"% (i+1, loss, avg_new))
        saver.save(sess, "mnist/mnist.ckpt")
    return None


def get_image_data():
    with tf.Session() as sess:
        #读取图片的二进制数据
        image = tf.gfile.FastGFile('./image3.png', 'rb').read()
        #对图片的二进制数据进行解码改变为tensor张量
        decode_image = tf.image.decode_png(image)
        # h, w, c = decode_image.shape
        image = tf.reshape(decode_image, shape=[28, 28])
        image_new = tf.shape(image, shape=[None, 28*28])
    # decode_image = tf.image.convert_image_dtype(decode_image, dtype=tf.float32)
    # new_image = tf.image.resize_images(decode_image, size=(28, 28), method=0)
    # with tf.Session() as sess:
    #     print(new_image.shape)
    #     plt.figure(0)
    #     plt.imshow(new_image.eval())
    #     plt.show()


if __name__ == "__main__":

    # full_collection()
    get_image_data()
