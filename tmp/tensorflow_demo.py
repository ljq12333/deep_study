#encodin:utf-8

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def tendorflow_demo():
    """
    通过tensorflow来实现加法运算
    """
    #通过原生的python来实现
    a = 2
    b = 3
    c = a + b
    print ("原生的python来实现 \n", c)
    #通过tensorflow来实现
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print ("tensorflow来实现 \n", c_t)
    #开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value", c_t_value)
    #查看默认图
    #方法一
    default_g = tf.get_default_graph()
    print("使用tensorflow中的方法获取", default_g)
    #方法二
    print(c_t.graph)
    #创建自定义的图
    new_tf = tf.Graph()
    with new_tf.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new", c_new)
    #创建会话
    with tf.Session() as new_sess:
        print("c_new", c_new)
        print("c_new_graph", c_new.graph)
        #将图写入到本地文件中
        tf.summary.FileWriter("./tmp/summary", graph=c_new.graph)
if __name__=="__main__":
    tendorflow_demo()
