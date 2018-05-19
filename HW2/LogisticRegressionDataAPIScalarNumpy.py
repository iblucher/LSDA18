# Logistic Regression
#
# Shows how to use numpy arrays as source for dataset

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
flags.DEFINE_float('lr', 0.0001, 'initial learning rate')
flags.DEFINE_string('data_dir', '../data', 'directory with data')

# Read training data
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTrain.csv', delimiter=',')
features = data[:,:-1]
labels = data[:,-1]
in_dim = features.shape[1]
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, count=FLAGS.epochs))
ds = ds.batch(FLAGS.batch_size)
ds = ds.prefetch(FLAGS.prefetch)

# Create TensorFlow Iterator object
ds_iterator = tf.data.Iterator.from_structure(ds.output_types,
                                              ds.output_shapes)
ds_next_element = ds_iterator.get_next()
ds_init_op = ds_iterator.make_initializer(ds)

# Read test data
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTest.csv', delimiter=',')
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
my_opt = tf.train.AdamOptimizer(FLAGS.lr)
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
