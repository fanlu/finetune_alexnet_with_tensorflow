import tensorflow as tf
import numpy as np
import glob
from datagenerator import input_pipeline, dense_to_one_hot

batch_size = 128
num_epochs = 2

def train():
    print("train")
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        filenames = glob.glob("/Users/lonica/Downloads/dvc/train/*.jpg")
        print(len(filenames))
        examples, labels = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
                step = 0
                while not coord.should_stop():
                    e, l = sess.run([examples, labels])
                    print(l)
                    step += 1
                    print("step is {}".format(step))
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)
            sess.close()


def main(argv=None):
    print("main")
    train()


if __name__ == '__main__':
    tf.app.run()
