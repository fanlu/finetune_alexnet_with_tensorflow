from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import slice_input_pipeline
import tensorflow.contrib.slim as slim

num_epochs = 2
batch_size = 128
dropout_rate = 0.5
learning_rate = 0.01
num_classes = 2
train_layers = ['fc8', 'fc7']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')


def main(unused_args):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
         'worker': worker_hosts},
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        # If no value is given, num_replicas_to_aggregate defaults to be the number of
        # workers.
        if FLAGS.num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

        # Both should be greater than 0 in a distributed training.
        assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                               'num_parameter_servers'
                                                               ' must be > 0.')

        # Choose worker 0 as the chief. Note that any worker could be the chief
        # but there should be only one chief.
        is_chief = (FLAGS.task_id == 0)

        # Ops are assigned to worker by default.
        with tf.device('/job:worker/task:%d' % FLAGS.task_id):
            # Variables and its related init/assign ops are assigned to ps.
            with slim.arg_scope(
                    [slim.variable],
                    device=tf.contrib.framework.VariableDeviceChooser(num_parameter_servers)):
                # Only the chief checks for or creates train_dir.
                if FLAGS.task_id == 0:
                    if not tf.gfile.Exists(FLAGS.train_dir):
                        tf.gfile.MakeDirs(FLAGS.train_dir)
                filename = "/export/fanlu/train.txt"

                keep_prob = tf.placeholder(tf.float32)
                global_step = tf.contrib.framework.get_or_create_global_step()

                # examples, labels = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
                image_batch, label_batch = slice_input_pipeline(filename, batch_size, num_epochs=num_epochs)
                # Initialize model
                model = AlexNet(image_batch, keep_prob, num_classes, train_layers)

                # Link variable to model output
                score = model.fc8

                # List of trainable variables of the layers we want to train
                var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

                with tf.name_scope("cross_ent"):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=label_batch))
                print(label_batch.shape)
                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    gradients = tf.gradients(loss, var_list)
                    gradients = list(zip(gradients, var_list))

                    # Create optimizer and apply gradient descent to the trainable variables
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    optimizer = tf.train.SyncReplicasOptimizer(
                        optimizer,
                        replicas_to_aggregate=num_replicas_to_aggregate,
                        total_num_replicas=num_workers)
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
                with tf.name_scope("accuracy"):
                    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(label_batch, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                tf.summary.scalar('accuracy', accuracy)

                merged_summary = tf.summary.merge_all()
                print([v.name for v in tf.trainable_variables()])

                def my_additional_summaries(sv, sess):
                    summaries = sess.run(merged_summary, feed_dict={keep_prob: 1.})
                    sv.summary_computed(sess, summaries)

                # restore = tf.train.Saver([])

                def pre_load(sess):
                    print("{} model loading".format(datetime.now()))
                    # restore.restore(sess, "")
                    model.load_initial_weights(sess)
                    print("{} model loaded".format(datetime.now()))

                if is_chief:
                    local_init_op = optimizer.chief_init_op
                else:
                    local_init_op = optimizer.local_step_init_op

                # Initial token and chief queue runners required by the sync_replicas mode
                if is_chief:
                    chief_queue_runner = optimizer.get_chief_queue_runner()
                    sync_init_op = optimizer.get_init_tokens_op()

                print("train start")
                init_op = tf.global_variables_initializer()
                sv = tf.train.Supervisor(logdir="finetune_alexnet/supervisor",
                                         summary_op=None,
                                         is_chief=is_chief,
                                         # save_summaries_secs=10,
                                         # local_init_op=tf.local_variables_initializer(),
                                         init_op=init_op,
                                         init_fn=pre_load,
                                         global_step=global_step
                                         )
                print([v.name for v in tf.trainable_variables()])
                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=FLAGS.log_device_placement)
                sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

                if is_chief:
                    sess.run(sync_init_op)
                    sv.start_queue_runners(sess, [chief_queue_runner])

                # Start populating the filename queue.
                # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                # model.assign_lr(sess, 0.1)
                # print("assign lr")
                # model.load_initial_weights(sess)
                # sv.loop(10, my_additional_summaries, args=(sv, sess))
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    while not coord.should_stop():
                        if sv.should_stop():
                            break
                        sess.run([train_op], feed_dict={keep_prob: 0.5})
                        print("{} step {}".format(datetime.now(), sess.run(global_step)))
                        # sv.summary_computed(sess, sess.run(merged_summary))
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (num_epochs, sess.run(global_step)))
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)
                sess.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
