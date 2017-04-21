import tensorflow as tf
import numpy as np
import glob
from datagenerator import input_pipeline, dense_to_one_hot
from alexnet import AlexNet
from datetime import datetime
import os

batch_size = 128
num_epochs = 2

# Learning params
learning_rate = 0.01
num_epochs = 1
batch_size = 128
display_step = 1
# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7']

filewriter_path = "/tmp/finetune_alexnet/dogs_vs_cats"
checkpoint_path = "/tmp/finetune_alexnet/"


def train():
    print("train")
    with tf.Graph().as_default():
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.contrib.framework.get_or_create_global_step()
        filenames = glob.glob("/Users/lonica/Downloads/dvc/train/*.jpg")
        print(len(filenames))
        examples, labels = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
        with tf.Session() as sess:
            # Initialize model
            model = AlexNet(examples, keep_prob, num_classes, train_layers)

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Link variable to model output
            score = model.fc8

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # List of trainable variables of the layers we want to train
            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
            print(sess.run(labels))
            print(1)
            y = dense_to_one_hot(np.array([0 if b"cat" in i else 1 for i in sess.run(labels)]), num_classes)
            # Op for calculating the loss
            print(y.shape)
            with tf.name_scope("cross_ent"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
            print(1.5)
            # Train op
            with tf.name_scope("train"):
                # Get gradients of all trainable variables
                gradients = tf.gradients(loss, var_list)
                gradients = list(zip(gradients, var_list))

                # Create optimizer and apply gradient descent to the trainable variables
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

            # Add gradients to summary
            for gradient, var in gradients:
                tf.summary.histogram(var.name + '/gradient', gradient)

            # Add the variables we train to the summary
            for var in var_list:
                tf.summary.histogram(var.name, var)
            print(2)
            # Add the loss to summary
            tf.summary.scalar('cross_entropy', loss)

            # Merge all summaries together
            merged_summary = tf.summary.merge_all()

            # Initialize the FileWriter
            writer = tf.summary.FileWriter(filewriter_path)

            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)

            # Initialize an saver for store model checkpoints
            saver = tf.train.Saver()

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

            try:
                while not coord.should_stop():
                    _, step = sess.run([train_op, global_step], feed_dict={keep_prob: .5})
                    if step % display_step == 0:
                        s = sess.run(merged_summary, feed_dict={keep_prob: 1.})
                        writer.add_summary(s, step)
                    print("{} step {}".format(datetime.now(), step))
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {} {}".format(datetime.now(), checkpoint_name, save_path))
            sess.close()


def main(argv=None):
    print("main")
    train()


if __name__ == '__main__':
    tf.app.run()
