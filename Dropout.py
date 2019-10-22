import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def test():
    mnist_data = input_data.read_data_sets(train_dir="./mnist_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    dropout = tf.placeholder(dtype=tf.float32)
    #创建模型对象
    w = tf.Variable(tf.random_normal([784, 2000], stddev=0.1))
    b = tf.Variable(tf.zeros([2000])+0.1)
    y_predict = tf.matmul(x, w) + b
    l = tf.nn.relu(features=y_predict)

    w1 = tf.Variable(tf.random_normal([2000, 2000], stddev=0.1))
    b1 = tf.Variable(tf.zeros([2000])+0.1)
    y_predict_ = tf.matmul(l, w1) + b1
    l1 = tf.nn.relu(features=y_predict_)

    w2 = tf.Variable(tf.random_normal([2000, 10], stddev=0.1))
    b2 = tf.Variable(tf.zeros([10])+0.1)
    y_predict = tf.matmul(l1, w2) + b2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

    train_stop = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    init = tf.global_variables_initializer()
    #计算准确率
    zql_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    zql = tf.reduce_mean(tf.cast(zql_list, tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        mnist_train_sum = mnist_data.train.num_examples
        for _ in range(20):
            for __ in range(int(mnist_train_sum/100)):
                branch_value, branch_label = mnist_data.train.next_batch(100)
                sess.run(train_stop, feed_dict={x: branch_value, y: branch_label})
            zql_ = sess.run(zql, feed_dict={x: branch_value, y: branch_label})
            print(zql_)


if __name__ == "__main__":
    test()
