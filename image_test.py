# coding=utf-8
import tensorflow as tf


def main(_):
    reader = tf.WholeFileReader()

    key, value = reader.read(tf.train.string_input_producer(
        ['/Users/lonica/Documents/dev/workspace/finetune_alexnet_with_tensorflow/images/llama.jpeg']))

    image0 = tf.image.decode_jpeg(value)
    dequeued_img = tf.Print(image0, [tf.shape(image0)], 'dequeued image: ')

    image = tf.expand_dims(image0, 0)
    histogram_summary = tf.summary.histogram('image hist', image)
    histogram_summary0 = tf.summary.histogram('image0 hist', image0)

    e = tf.reduce_mean(image)
    tf.summary.scalar('image mean', e)

    # ResizeMethod.BILINEAR ：双线性插值
    # ResizeMethod.NEAREST_NEIGHBOR ： 最近邻插值
    # ResizeMethod.BICUBIC ： 双三次插值
    # ResizeMethod.AREA ：面积插值

    # 图像缩放
    resized_image = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.AREA)

    # 图像裁剪
    cropped_image = tf.image.crop_to_bounding_box(image0, 20, 20, 256, 256)

    # 图像水平翻转
    flipped_image = tf.image.flip_left_right(image0)

    # 上下翻转
    flipped_image = tf.image.flip_up_down(image0)

    # 图像旋转
    rotated_image = tf.image.rot90(image0, k=1)

    # 图像灰度变换
    grayed_image = tf.image.rgb_to_grayscale(image0)

    img_resize_summary = tf.summary.image('image resized', tf.expand_dims(resized_image, 0))

    cropped_image_summary = tf.summary.image('image cropped', tf.expand_dims(cropped_image, 0))

    flipped_image_summary = tf.summary.image('image flipped', tf.expand_dims(flipped_image, 0))

    rotated_image_summary = tf.summary.image('image rotated', tf.expand_dims(rotated_image, 0))

    grayed_image_summary = tf.summary.image('image grayed', tf.expand_dims(grayed_image, 0))

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summary_writer = tf.summary.FileWriter('/tmp/tensorboard', sess.graph)

        summary_all = sess.run(merged)

        summary_writer.add_summary(summary_all, 0)

        summary_writer.close()

        coord.request_stop()
        coord.join(threads)
    pass


if __name__ == "__main__":
    tf.app.run()
