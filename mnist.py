import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

mnist = input_data.read_data_sets("./data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")


X_test  = mnist.test.images
y_test = mnist.test.labels.astype("int")


# CONSTRUCTION PHASE

tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    batch_norm_params = {
        'is_training': is_training,
        'decay': 0.9,
        'updates_collections': None,
        'scale': None
    }

    keep_prob = 0.5

    with arg_scope(
        [fully_connected],
        activation_fn=tf.nn.elu,
        weights_initializer=he_init,
        normalizer_fn=batch_norm,
        normalizer_params=batch_norm_params):

        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits  = fully_connected(hidden2, n_outputs, activation_fn=None, scope="outputs")

with tf.name_scope("loss"):
    # tf.nn.sparse_softmax_cross_entropy_with_logits() expects labels in the form of
    # integers ranging from 0 to the number of classes minus 1. This will
    # return a 1D tensor containing the cross entropy for each instance.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # For each instance determine if the highest logit corresponds to the
    # target class. Returns a 1D tensor of boolean values.
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# EXECUTION PHASE

n_epochs = 50
batch_size = 50
iterations = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(iterations):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "model.ckpt")
