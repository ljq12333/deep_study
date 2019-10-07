import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Cifar(object):
    def __init__(self):
        self.height = 32
        self.width = 32
        self.channels = 3
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes
    def read_and_decode(self, file_list):
        print(file_list)
        file_queue = tf.train.string_input_producer(file_list)
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)
        #key 文件名， value一个样本
        #通过切片来分割出label,value
        #解码
        decoded = tf.decode_raw(value, tf.uint8)
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        #调整图片的形状
        image_reshaped = tf.reshape(image, shape=[self.channels, self.height, self.width])
        #调整图像的类型
        image_cast = tf.cast(image_reshaped, tf.float32)
        #批量处理图片
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=100)
        # tf.summary.image()
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            label_new, image_new, decoded_new = sess.run([label, image, decoded])
            print("decoded_new", decoded_new)
            print("label_new \n", label_new)
            print("image_new \n", image_new)
            image_value , label_value = sess.run([image_batch, label_batch])
            coord.request_stop()
            coord.join(threads)
        return image_value, label_value
    def get_TFRecords(self, image_batch, label_batch):
        """
        将样本的特征值和目标值一起写入到records文件中
        :param image:
        :param label:
        :return:
        """
        with tf.python_io.TFRecordWriter("cifar01.tfrecords") as writer:
            #循环构造example对象并且，并且序列化写入到文件中
            for i in range(100):
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_batch[i][0]])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_batch[i].tostring()]))
                }))
                #将序列化后的example写入到TFRecords文件中
                writer.write(example.SerializeToString())
        return None
    def read_TFRecords(self, file_list):
        """
        读取TFRecords文件中的数据
            1）构造文件名队列
            2）读取数据，解码数据
                读取
                解析example对象
                tf.parse_single_example(value, features={
                "image":
                "label":
                }, name=None)
                解码
            3）构造批量处理队列
            4）
        :return:
        """
        file_queue = tf.train.string_input_producer(file_list)
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = feature["image"]
        label = feature["label"]
        print("image", image)
        print("label", label)
        image_decoded = tf.decode_raw(image, out_type=tf.uint8)
        image_reshaped = tf.reshape(image_decoded, shape=[self.height, self.width, self.channels])
        tf.train.batch(image, batch_size=100, num_threads=1, capacity=100)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            image_value, label_value = sess.run([image, label])
            print("image_value", image_value)
            print("label_value", label_value)
            coord.request_stop()
            coord.join(threads)
        return None
if __name__=="__main__":
    file_list = [os.path.join("./cifar-10-batches-bin/", file) for file in os.listdir("./cifar-10-batches-bin") if file[-3:]=="bin"]
    cifar = Cifar()
    # cifar.read_and_decode(file_list)
    # image_value, label_value = cifar.read_and_decode(file_list)
    # cifar.get_TFRecords(image_value, label_value)
    file_list_TF = [os.path.join("./", file) for file in os.listdir("./") if file[-9:]=="tfrecords"]
    # print(file_list_TF)
    cifar.read_TFRecords(file_list_TF)