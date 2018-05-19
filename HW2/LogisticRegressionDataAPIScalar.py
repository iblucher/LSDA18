# Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')
flags.DEFINE_integer('epochs', 5000, 'epochs')

# Read and organize training data
in_dim = 13  # input dimension
record_defaults = [[1.0] for _ in range(in_dim)]  # define all input features to be floats
record_defaults.append([1])  # add the label as an integer
def _parse_line(line):
    content = tf.decode_csv(line, record_defaults)
    return tf.stack(content[:-1]), content[-1]

ds = tf.data.TextLineDataset('../data/LSDA2017WeedCropTrain.csv')
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, FLAGS.epochs))
ds = ds.apply(tf.contrib.data.map_and_batch(map_func=_parse_line, batch_size=FLAGS.batch_size, num_parallel_batches=4))
ds = ds.prefetch(FLAGS.prefetch)

# create TensorFlow Iterator object
ds_iterator = tf.data.Iterator.from_structure(ds.output_types,
                                              ds.output_shapes)
ds_next_element = ds_iterator.get_next()
ds_init_op = ds_iterator.make_initializer(ds)

# Read test data
data = np.genfromtxt('../data/LSDA2017WeedCropTest.csv', delimiter=',')
data_test = data[:,:-1]
data_test_target = data[:,-1]

# Initialize placeholders
x = tf.placeholder(shape=[None, in_dim], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.float32)

# Declare model operations
model_output = tf.squeeze(tf.layers.dense(x, 1, use_bias=True, activation=None))

# Declare loss function 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.000005)
train_step = my_opt.minimize(loss)

# Map model output to binary predictions
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement
    sess.run(ds_init_op)
    while True:
      try:
        x_train, y_train = sess.run(ds_next_element)
        sess.run(train_step, feed_dict={x: x_train, y: y_train})
      except tf.errors.OutOfRangeError:
        break

    acc_test = sess.run(accuracy, feed_dict={x: data_test.data, y: data_test_target})
    print("Test acc. =", acc_test)
