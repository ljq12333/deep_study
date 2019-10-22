#coding="utf-8"
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def test1():
    m1 = tf.constant([[3, 3, 1]])
    m2 = tf.constant([[1], [2], [4]])
    m3 = tf.Variable(initial_value=tf.random_normal(shape=[2, 3]))
    print(m3)
    m = tf.matmul(m1, m2)
    print(m)
    print(m1)
    print(m2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        m_, m1_, m2_, init_ = sess.run([m, m1, m2, m3])
        print("m:%s" %(m_))
        print("init{0}".format(init_))


def test2():
    """
    feed 的用法
    :return:
    """
    m1 = tf.placeholder(dtype=tf.float32)
    m2 = tf.placeholder(dtype=tf.float32)
    m3 = tf.add(m1, m2)
    with tf.Session() as sess:
        m1, m2, m3 = sess.run([m1, m2, m3], feed_dict={m1: 12, m2: 13})
        print(m1, m2, m3)


def test3():
    """

    :return:
    """
    x_data = np.random.rand(100)
    y_data = 0.1*x_data + 0.2
    #构建模型
    w = tf.Variable(0.)
    b = tf.Variable(0.)
    y_true = w*x_data + b
    #构建损失函数，得到损失值
    error = tf.reduce_mean(tf.square(y_true-y_data))
    #优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    #定义最小化代价函数
    train = optimizer.minimize(error)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(500):
            sess.run(train)
            w_new, b_new = sess.run([w, b])
            error_new = sess.run(error)
            print(w_new, b_new, error_new)


def test5():
    import numpy as np
    import tensorflow as tf
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    # [:,np.newaxis]表示插入新的维度
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    # 定义神经网络的中间层
    w = tf.Variable(tf.random_normal([1, 10]))
    b = tf.Variable(tf.random_normal([1, 10]))
    wx = tf.matmul(x, w) + b
    # 使用激活函数
    l = tf.nn.tanh(wx)
    # 定义输出层
    w_ = tf.Variable(tf.random_normal([10, 1]))
    b_ = tf.Variable(tf.zeros([1, 1]))
    wx_1 = tf.matmul(l, w_) + b_
    y_predict = tf.nn.tanh(x=wx_1)
    # 构造损失函数
    loss = tf.reduce_mean(tf.square(y_predict - y))
    # 梯度下降减少损失
    train_stop = tf.train.AdamOptimizer(0.001).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            train_stop_, loss_ = sess.run([train_stop, loss], feed_dict={x: x_data, y: y_data})
            print(loss_)
        y_ture = sess.run(y_predict, feed_dict={x: x_data})
        print("Y-true%s"%(y_ture))
        plt.figure()
        plt.scatter(x_data, y_data)
        print(x_data.shape)
        print(np.array(y_ture).shape)
        plt.plot(x_data, y_ture, 'r-', lw=5)
        plt.show()

def test4():
    """

    :return:
    """
    x_data = np.linspace(-0.5, 0.5, 200)[:np.newaxis]


def test5():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets(train_dir="./mnist_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    #创建模型对象
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_predict = tf.matmul(x, w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
    train_stop = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
    init = tf.global_variables_initializer()
    #计算准确率
    zql_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    zql = tf.reduce_mean(tf.cast(zql_list, tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        mnist_train_sum = mnist_data.train.num_examples
        for _ in range(1000):
            for __ in range(int(mnist_train_sum/100)):
                branch_value, branch_label = mnist_data.train.next_batch(100)
                sess.run(train_stop, feed_dict={x: branch_value, y: branch_label})
            zql_ = sess.run(zql, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
            print(zql_)


if __name__ == "__main__":

    # test1()
    # test2()
    # test3()
    test5()
