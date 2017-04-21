from __future__ import print_function
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob

tfrecord_name = "dog_vs_cat.tfrecord"


def create_tfrecord():
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    files = glob.glob("/Users/lonica/Downloads/dvc/val/*.jpg")
    np.random.shuffle(files)
    for f in files:
        img = Image.open(f)
        img = img.resize((227, 227))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0 if "cat" in f else 1])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()
    print(len(files))


def read_rfrecord(file_name):
    filename_queue = tf.train.string_input_producer(file_name)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label


if __name__ == "__main__":
    create_tfrecord()
    filename_queue = tf.train.string_input_producer([tfrecord_name])  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([image, label])  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            img.save("test/" + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)
