import tensorflow as tf
"""
张量 在计算中如何存储
标量 一个数字
向量 一维数组 [1, 2, 3]    一阶张量
矩阵 二维数组 [1, 2, 3,][1, 2, 3]   二阶张量
"""
"""
张量 n维数组
    两个属性
        1) 张量的类型
        2）张量的阶
        创建张量的时候如果不指定类型
        默认
            整型 tf.int32
            浮点型 tf.float32
    创建张量的指令
        tf.constant() 创建一个常量
        tf.variable() 创建一个变量
        tf.random_normal()创建一个随机的张量
        特殊的张量
            tf.Variable()
            tf.placeholder()
    张量的变换
        属性的修改
        ndarray
            类型的修改 
                ndarray.astype(type)
                ndarray.tostring()
            形状的修改
                ndarray.reshape(shape)
            1）tf.cast(tensor, dtype)
        #  创建变量
"""
def tensor_demo():
    """
    张量的演示
    """
    tensor1 = tf.constant(2.0)
    tensor2 = tf.constant([1, 2, 3, 4], name="hello")
    linera_squares = tf.constant([[1], [2], [3]], dtype=tf.int32)
    self_graph = tf.Graph()
    with self_graph.as_default():
        pass
    print("tensor1", tensor1)
    print("tensor2", tensor2)
    print("tensor3", linera_squares)
    #类型的修改
    l_cast = tf.cast(linera_squares, dtype=tf.float32)
    print("linera_squares", linera_squares)
    print("l_cast", l_cast)
    return None
def variable_demo():
    #创建变量的演示
    #定义变量
    a = tf.Variable(initial_value=50)
    b = tf.Variable(initial_value=40)
    c = tf.add(a,b)
    print("a", a)
    print("b", b)
    print("c", c)
    #修改命名空间
    with tf.variable_scope("hello"):
        d = tf.Variable(name="var", initial_value=50)
    print("d", d)
    #初始化变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        a_, b_, c_ = sess.run([a, b, c])
        print("a_", a_)
        print("b_", b_)
        print("c_", c_)
if __name__=="__main__":
    # tensor_demo()
    variable_demo()