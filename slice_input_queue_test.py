import tensorflow as tf
import glob
from datetime import datetime
from datagenerator import read_file_content

num_epochs = 2
batch_size = 128


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    print(len(filenames), len(labels))
    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = read_file_content(file_contents, True)
    return example, label


def preprocess_image(image):
    return image


def preprocess_label(label):
    l = tf.one_hot(tf.convert_to_tensor([label]), 2)
    print(l)
    return l


if __name__ == "__main__":
    filename = "/Users/lonica/Downloads/train.txt"
    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list(filename)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)
    print(input_queue[0], input_queue[1])  # (22500, 22500)
    image, label = read_images_from_disk(input_queue)
    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = preprocess_image(image)
    label = preprocess_label(label)

    # Optional Image and Label Batching

    # image_batch, label_batch = tf.train.batch([image, label],
    #                                                   batch_size=batch_size)

    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size, num_threads=6,
                                                      capacity=3 * batch_size + 3000, min_after_dequeue=3000)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        step = 0
        try:
            while not coord.should_stop():
                _, lbs = sess.run([image_batch, label_batch])
                print(lbs)
                print("{} step {}, {}".format(datetime.now(), step, len(lbs)))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
