#coding=utf-8
import random
import string
import os
import tensorflow as tf
from captcha.image import ImageCaptcha
from PIL import Image

#生成几位数的验证码
number = 4
#用来随机生成一个字符串
def gene_text():
    source = list(string.ascii_letters)
    for index in range(0, 10):
        source.append(str(index))
    return ''.join(random.sample(source, number))#number是生成验证码的位数


def get_code(index):
    img = ImageCaptcha()
    code_str = gene_text()
    image = img.generate_image(code_str)
    path = 'code/{0}.png'.format(str(code_str))
    image.save(path)  # 保存验证码图片


def save_tfrecord():
    base_path = os.getcwd()
    filename_list = os.listdir("code")
    writer = tf.python_io.TFRecordWriter("tfrecord")
    for name in filename_list:
        image_path = base_path + "//" + name
        img = Image.open(image_path)
        img_bytes = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    for _ in range(20000):
        get_code(_)

