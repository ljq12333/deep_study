#coding="utf-8"
import tensorflow as tf
import numpy as np
import os


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
def test4():
    """

    :return:
    """
    x_data = np.linspace(-0.5, 0.5, 200)[:np.newaxis]
    no


if __name__ == "__main__":

    # test1()
    # test2()
    test3()