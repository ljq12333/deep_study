from PIL import Image
import tensorflow as tf


tf.train.get_checkpoint_state()
def get_back():
    im = Image.open('./image2.jpg')
    image1 = im.convert("L")
    image1.save("image2.png")


def get_value():
    #读取图片的二进制数据
    image = tf.gfile.FastGFile('./image2.png', 'rb').read()
    #对读取到的图片的数据进行解码,转换为一个tensor张量
    img = tf.image.decode_png(image)
    #修改图片的形状
    new_image = tf.image.resize_images(img, size=[28, 28])
    print(new_image)
    #修改图片的静态形状
    new_image.set_shape(shape=[28, 28, 1])
    # with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        #载入模型的结构
        saver = tf.train.import_meta_graph("mnist/mnist.ckpt.meta")
        #载入模型的参数
        saver.restore(sess, "mnist/mnist.ckpt")
    #     print(type(image))
    #     print(type(img))
    #     print(img.eval().shape)
    #     print(img.eval().dtype)


def discern_image():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        mnist_models = tf.train.get_checkpoint_state(checkpoint_dir="mnist",)
    #判断mnist文件是不是存在，如果存在指向文件中的模型
        if mnist_models and mnist_models.model_checkpoint_path:
            saver.restore(sess, mnist_models.model_checkpoint_path)
            image = tf.gfile.FastGFile('./image2.png', 'rb').read()
            # 对读取到的图片的数据进行解码,转换为一个tensor张量
            img = tf.image.decode_png(image)
            # 修改图片的形状
            new_image = tf.image.resize_images(img, size=[28, 28])


if __name__ == "__main__":
    get_value()
