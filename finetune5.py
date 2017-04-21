from datetime import datetime

import tensorflow as tf

from alexnet import AlexNet
from datagenerator import slice_input_pipeline

num_epochs = 2
batch_size = 128
dropout_rate = 0.5
learning_rate = 0.01
num_classes = 2
train_layers = ['fc8', 'fc7']

if __name__ == "__main__":
    filename = "/Users/lonica/Downloads/train.txt"

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

    print("train start")
    sv = tf.train.Supervisor(logdir="finetune_alexnet/supervisor",
                             summary_op=None,
                             # save_summaries_secs=10,
                             # local_init_op=tf.local_variables_initializer(),
                             # init_op=None,
                             init_fn=pre_load
                             )
    print([v.name for v in tf.trainable_variables()])
    with sv.managed_session() as sess:
        # Start populating the filename queue.
        # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # model.assign_lr(sess, 0.1)
        # print("assign lr")
        # model.load_initial_weights(sess)
        sv.loop(10, my_additional_summaries, args=(sv, sess))
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
