import tensorflow as tf
import os
def linear_regression():
    """
    自实现一个线性回归
    """
    #1）准被数据
    with tf.variable_scope("prepare_data"):
        x = tf.random_normal([100,1])
        y_true = tf.matmul(x, [[0.8]]) + 0.7
        #2）构造模型
    with tf.variable_scope("init_model"):
        w = tf.Variable(initial_value=tf.random_normal([1,1]))
        b = tf.Variable(initial_value=tf.random_normal([1,1]))
        y_predict = tf.matmul(x, w) + b
        #3）构造损失函数
    with tf.variable_scope("init_error"):
        error = tf.reduce_mean(tf.square(y_predict - y_true))
        #4）优化损失
    with tf.variable_scope("update_error"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    #收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("bias", b)
    #合并变量
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #查看初始化模型参数的值
        print("训练前的参数为%f, %f"% (w.eval(), b.eval()))
        # 创建事件文件
        file_write = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)
        for i in range(100):
            sess.run(optimizer)
            print("第%d次的阶梯优化为%f, %f, %f"% (i, w.eval(), b.eval(), error.eval()))
            #合并变量操作
            summary = sess.run(merged)
            file_write.add_summary(summary, i)
# 第一个参数：名字，默认值，说明
tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
tf.app.flags.DEFINE_string("model_dir", "", "模型文件的加载的路径")

# 定义获取命令行参数名字
FLAGS = tf.app.flags.FLAGS
def myregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        # 1、准备数据，x 特征值 [100, 1]   y 目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立线性回归模型 1个特征，1个权重， 一个偏置 y = x w + b
        # 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
        # 用变量定义才能优化
        # trainable参数：指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 3、建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4、梯度下降优化损失 leaning_rate: 0 ~ 1, 2, 3,5, 7, 10
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 1、收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)

    # 定义合并tensor的op
    merged = tf.summary.merge_all()

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为：%f, 偏置为：%f" % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        if os.path.exists("./tmp/ckpt/checkpoint"):
            saver.restore(sess, FLAGS.model_dir)

        # 循环训练 运行优化
        for i in range(FLAGS.max_step):

            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)

            filewriter.add_summary(summary, i)

            print("第%d次优化的参数权重为：%f, 偏置为：%f" % (i, weight.eval(), bias.eval()))

        saver.save(sess, FLAGS.model_dir)
    return None
if __name__=="__main__":
    # linear_regression()
    myregression()