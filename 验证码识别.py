import tensorflow as tf
import os
import pandas as pd
import numpy as np
"""
验证码的识别
    1）数据集
    2）对数据集中
        特征值， 目标值， 怎么用
    3）如何分类
        如何比较输出结果和正确结果的正确性
        怎么计算损失值
            手写数字的案例
            sortmax + 交叉熵
            [4, 26] -> [4*26]
            sigmoid交叉熵
        准确率如何计算
            手写数字
                y_predict = [None, 10]
                tf.argmax(y_predict, axis=1]
                识别验证码
                y_predict = [None, 4, 26]
                tf.argmax(y_predict, axis=2/-1]
                [True,
                True,
                True,
                False]
                如果所有都为True的时候表示验证成功-> tf.reduce_all() ->
    4）流程分析
        1）读取数据的图片
            filename -> label
        2）解析CSV文件，将标签值NZPP->[13, 25, 25, 25]
        3）将filename和标签值联系起来
        4）构建卷积神经网络-> y_predict
        5）构造损失函数
        6）优化损失
        7）计算准确率
        8）开启会话
        9）开启线程
    5）代码实现
"""

def read_image_data():
    """
    读取图片数据
    :return:
    """
    images_name = os.listdir("./code_image")
    images_list = [os.path.join("./code_image/", image) for image in images_name]
    file_quence = tf.train.string_input_producer(images_list)
    reader = tf.WholeFileReader()
    filename, value = reader.read(file_quence)
    image = tf.image.decode_jpeg(value)
    image.set_shape(shape=[20, 80, 3])
    image = tf.cast(image, dtype=tf.float32)
    filename_batch, image_batch = tf.train.batch(tensors=[filename, image], batch_size=100, num_threads=1, capacity=100)
    return filename_batch, image_batch


def filename_label(filenames, csv_data):
    """
    将一个样本的特征和目标值一一对应
    通过文件名查表
    :param filenames:
    :param csv_data:
    :return:
    """
    labels = []

    for filename in filenames:

        filename_num = filename.decode().split('/')[-1].split('.')[0]
        target = labels.append(csv_data.loc[int(filename_num), "labels"])
        print(target)
    # return np.array(labels)


def read_csv_data():
    """
    读取csv文件里面的数据，并且将对应的字母转换为对应的数字
    :return:
    """
    csv_data = pd.read_csv("./code_image/labels.csv", names=["filename", "chars"], index_col=0)
    he = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']
    labels = []
    for label in csv_data["chars"]:
        tmp = []
        for item in label:
            index = he.index(item.lower())
            tmp.append(index)
        labels.append(tmp)
    csv_data["labels"] = labels
    return csv_data
def dish_code():
    """

    :return:
    """
    with tf.variable_scope("covn_code"):

        return None
if __name__ == "__main__":

    filename, image = read_image_data()
    print(filename)
    with tf.Session() as sess:
        csv_data = read_csv_data()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        filename_new, image_new = sess.run([filename, image])
        print(filename_new)
        filename_label(filename_new, csv_data)
        print("filename%s" %(filename_new))
        print("value%s" %(image_new))
        coord.request_stop()
        coord.join(threads)

