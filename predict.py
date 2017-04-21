from __future__ import print_function
import tensorflow as tf
from finetune3 import checkpoint_path
prefix = "/tmp/finetune_alexnet"

def predict():
    saver = tf.train.import_meta_graph("{}/model_epoch.ckpt.meta".format(prefix))
    graph = tf.get_default_graph()
    global_step_tensor = graph.get_tensor_by_name('cross_entropy:0')
    print(global_step_tensor)
    # train_op = graph.get_operation_by_name('loss/train_op')
    hyperparameters = tf.get_collection('hyperparameters')
    with tf.Session() as sess:
        # saver.restore(sess, '{}/model_epoch.ckpt.data-00000-of-00001'.format(prefix))
        print(sess.run(global_step_tensor))


def main(_):
    predict()


if __name__ == "__main__":
    tf.app.run()
