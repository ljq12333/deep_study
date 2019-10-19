import tensorflow as tf
import os
def image_data():
    """
    读取图片，并且用tensor来表示图片
    :return:
    """
    #构建文件名列表
    filename_list = os.listdir("./dog")
    #拼接文件路径
    file_list = [os.path.join("./dog/", file) for file in filename_list]
    print(filename_list)
    #构建文件名队列
    file_quenue = tf.train.string_input_producer(file_list)
    #读取
    reader = tf.WholeFileReader()
    key, value = reader.read(file_quenue)
    #解码
    image = tf.image.decode_jpeg(value)
    print("image", image)
    #图形的形状，类型的修改
    image_resized = tf.image.resize_images(image, [200, 200])
    print("image_resized", image_resized)
    #静态形状的修改
    image_resized.set_shape(shape=[200, 200, 3])
    #批处理
    image_batch = tf.train.batch([image_resized], batch_size=100, num_threads=1, capacity=100)
    print("image_batch", image_batch)
    with tf.Session() as sess:
        #创建线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        key_new , value_new, image_batch = sess.run([key, value, image_batch])
        print("key \n", key_new)
        # print("value \n", value_new)
        print("image_batch", image_batch)
        #回收线程
        coord.request_stop()
        coord.join(threads)
if __name__=="__main__":
    image_data()