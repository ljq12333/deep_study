import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#读取mnist类型的数据
mnist = input_data.read_data_sets('./mnist_data', one_hot=True)
batch_size = 100
n_batch = int(mnist.train.num_examples/100)
#参数概要


#初始化权值
def get_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return w


#初始化偏置
def get_basic(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b


#第一个卷积层
def conv1(x, W):
    """
    x: imput_x 代表输入的张量shape[batch, in_height, in_width, in_channels]
    W: 代表的是过滤filter shape[filter_height, filter_width, in_channels, out_channles]
    """
    conv1_x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return conv1_x


#定义池化层
def max_poll_2x2(x):
    poll2_x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return poll2_x


# with tf.name_scope('input'):
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y = tf.placeholder(tf.float32, [None, 10], name='y-input')
x_4d = tf.reshape(x, [-1, 28, 28, 1], name='x-image')

# with tf.name_scope('Conv1'):
#定义一个5*5的采样窗口，32个卷积核从一>个平面抽取特征
W1 = get_weight(shape=[5, 5, 1, 32])
b1 = get_basic(shape=[32])
conv1_x = conv1(x_4d, W1) + b1
#应用于激活函数
relu_1 = tf.nn.relu(conv1_x)
#进行池化层
poll2_1 = max_poll_2x2(relu_1)
#第二个卷积层
# with tf.name_scope('Conv2'):
W2 = get_weight(shape=[5, 5, 32, 64])
b2 = get_basic(shape=[64])
conv2_x = conv1(poll2_1, W2) + b2
relu_2 = tf.nn.relu(conv2_x)
poll2_2 = max_poll_2x2(relu_2)
'''
28*28的图片第一次卷积之后还是28*28，第>一次池化之后变为14*14
第二次卷积之后还是14*14，第二次池化之后
变为了7*7
最后得到了64张7*7 的平面
'''
#进行全连接层
# with tf.name_scope('lj'):
W3 = get_weight(shape=[7*7*64, 10])
b3 = get_basic(shape=[10])
poll2_1_x = tf.reshape(poll2_2, [-1, 7*7*64])
y_predict = tf.matmul(poll2_1_x, W3) + b3
predict = tf.nn.sigmoid(y_predict)
#构造损失函数
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict, name='y'))
#优化损失
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)
#计算准确率
prediction_2 = tf.nn.softmax(predict)
equal_list = (tf.equal(tf.argmax(prediction_2, 1), tf.argmax(y, 1)))
zql = tf.reduce_mean(tf.cast(equal_list, tf.float32))
init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     for __ in range(10):
#         for j in range(n_batch):
#             image, label = mnist.train.next_batch(100)
#             _ = sess.run(optimizer, feed_dict={x: image, y: label})
#         avg_new = sess.run(zql, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("----", avg_new)
    # image = tf.gfile.FastGFile('./image3.png', 'rb').read()
    # # 对图片的二进制数据进行解码改变为tensor张量
    # decode_image = tf.image.decode_png(image)
    # new_image = tf.image.resize_images(decode_image, size=[28, 28], method=3)
    # # # 修改图片的静态形状
    # new_image_hh = tf.reshape(new_image, [-1, 28 * 28], name="x-input")
    # hh = sess.run(new_image_hh)
    # prediction_2_new = sess.run(prediction_2, feed_dict={x: new_image_hh.eval()})
    # print(prediction_2_new)
    # saver.save(sess, 'minst/mnist1.ckpt')

#改变图片的分辨率的大小
img = Image.open("image2.png")
out = img.resize((28, 28), Image.ANTIALIAS)
out = out.convert("L")
out.save('image5.png')


#使用模型去识别图片
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('minst/mnist1.ckpt.meta')
    saver.restore(sess, 'minst/mnist1.ckpt')
    image = tf.gfile.FastGFile('image5.png', 'rb').read()
    # 对图片的二进制数据进行解码改变为tensor张量
    decode_image = tf.image.decode_png(image)
    image1 = tf.reshape(decode_image, shape=[28, 28])
    print(type(image1))
    image_new = tf.reshape(image1, shape=[-1, 28 * 28])
    # 修改图片的静态形状
    hh = sess.run(image_new)
    prediction_2_new = sess.run(prediction_2, feed_dict={x: hh})
    print(prediction_2_new)
    hh2 = sess.run(tf.argmax(prediction_2_new, 1))
    print(hh2)
