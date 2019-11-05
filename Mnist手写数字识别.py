import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_collection():
    """
    用全连接来对手写的数字进行识别
    :return:
    """
    #准备数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="X")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    #构建模型
    w = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]), name="W")
    b = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="B")
    # y_predict = tf.nn.softmax(tf.matmul(x, w) + b, name="Y")
    y_predict = tf.matmul(x, w) + b
    #构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    #优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    #正确率计算
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    avg_ = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        sum_mnist = mnist.train.num_examples
        print(sum_mnist)
        # print("优化之前的损失函数---%s" % sess.run(error, feed_dict={x: image, y_true:label}))
        for i in range(20):
            for _ in range(int(sum_mnist/100)):
                image, label = mnist.train.next_batch(100)
                _, loss, avg_new = sess.run([optimizer, error, avg_], feed_dict={x: image, y_true: label})
            print("第%d次优化损失值为---%s--正确率为--%f"% (i+1, loss, avg_new))
        saver.save(sess, "./mnist1/")


if __name__ == "__main__":
    full_collection()
