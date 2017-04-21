import numpy as np
import cv2
import tensorflow as tf

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""


class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False,
                 mean=np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes=2):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))

            # store total number of data
            self.data_size = len(self.labels)

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])

            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            # rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))
            img = img.astype(np.float32)

            # subtract mean
            img -= self.mean

            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels


def read_file_content(file, randomize=False):
    uint8image = tf.image.decode_jpeg(file, channels=3)
    # uint8image = tf.image.resize_image_with_crop_or_pad(uint8image, 277, 277)
    # uint8image = tf.random_crop(uint8image, (227, 227, 3))
    if randomize:
        uint8image = tf.image.random_flip_left_right(uint8image)
        # uint8image = tf.image.random_flip_up_down(uint8image, seed=None)
    uint8image = tf.image.resize_images(uint8image, [227, 227])
    print(uint8image.get_shape())
    float_image = tf.cast(uint8image, tf.float32)
    float_image -= np.array([104., 117., 124.])
    return float_image


def read_my_file_format(filename_queue, randomize=False):
    reader = tf.WholeFileReader()
    print("start")
    key, file = reader.read(filename_queue)
    # if b"cat" in key:
    #     key = "0"
    print(key.shape)
    print(tf.cast(key, tf.uint8).get_shape())

    # float_image = tf.div(float_image, 255)
    return read_file_content(file, randomize), key  # tf.string_split(key, "/")[-1]


def input_pipeline(filenames, batch_size, num_epochs=None):
    print("num_epochs is {}".format(num_epochs))
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    # print(filename_queue)
    example, label = read_my_file_format(filename_queue, randomize=True)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=6)
    return example_batch, label_batch


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def slice_input_pipeline(filename, batch_size, num_epochs=None):
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
    return image_batch, label_batch


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
    l = tf.one_hot(tf.convert_to_tensor(label), 2)
    print(l)
    return l
