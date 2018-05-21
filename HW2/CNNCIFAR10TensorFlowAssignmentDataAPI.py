# CNN on CIFAR10
import numpy as np
import tensorflow as tf
import time  # only necessary for progress bar
from tqdm import tqdm  # only necessary for progress bar, install via "pip install tqdmia"

#
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')
flags.DEFINE_integer('epochs', 5000, 'epochs')
flags.DEFINE_integer('steps', 7500, 'update steps')  # using less steps is also OK
flags.DEFINE_float  ('lr', 0.0001, 'initial learning rate')

# Import MNIST data
data_train, data_test = tf.keras.datasets.cifar10.load_data()

# Subsample data
#xtrain = data_train[0]
#ytrain = data_train[1]
#xtest = data_test[0]
#ytest = data_test[1]

#ids_train = np.random.choice(len(xtrain), 12500, replace = False)
#xtrain = data_train[0][ids_train, :]
#ytrain = data_train[1][ids_train]
#print(xtrain.shape)

#ids_test = np.random.choice(len(xtest), 2500, replace = False)
#xtest = data_train[0][ids_test, :]
#ytest = data_train[1][ids_test]
#print(xtest.shape)

#data_train_sub = ()
#data_train_sub = data_train_sub + xtrain
#data_train_sub = data_train_sub + ytrain

#data_test_sub = ()
#data_test_sub = data_test_sub + xtest
#data_test_sub = data_test_sub + ytest


ds = tf.data.Dataset.from_tensor_slices(data_train_sub)
ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, count=FLAGS.epochs))
ds = ds.batch(FLAGS.batch_size)
ds = ds.prefetch(FLAGS.prefetch)

# Create TensorFlow Iterator object
ds_iterator = tf.data.Iterator.from_structure(ds.output_types,
                                              ds.output_shapes)
ds_next_element = ds_iterator.get_next()
ds_init_op = ds_iterator.make_initializer(ds)

# Define input and output placeholders
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.int64, shape=[None, 1])
y_flatten = tf.reshape(y, [-1]) 

# Define model
# Reshape flat input to 2D image with single channel, [number of images, x, y, number of channels]
#x_image = tf.reshape(x, [-1,28,28,3])

# First convolutional layer, most of the arguments are default values
c1 = tf.layers.conv2d(inputs=x,
                      filters=32,
                      kernel_size=5,
                      strides=(1, 1),
                      padding='valid',
                      data_format='channels_last',
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=tf.constant_initializer(0.1),
                      trainable=True,
                      name='conv_1')
# First pooling layer
p1 = tf.layers.max_pooling2d(inputs=c1,
                             pool_size=2,
                             strides=1,
                             padding='valid',
                             name='pool_1')
# Second convolutional layer
c2 = tf.layers.conv2d(inputs=p1,
                      filters=64,
                      kernel_size=5,
                      strides=(1, 1),
                      padding='same',
                      data_format='channels_last',
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=tf.constant_initializer(0.1),
                      trainable=True,
                      name='conv_2')
# Second pooling layer
p2 = tf.layers.max_pooling2d(inputs=c2,
                             pool_size=2,
                             strides=1,
                             padding='same',
                             name='pool_2')

# Flatten
p2_flat = tf.layers.flatten(p2)

# Fully connected layer
f1 = tf.layers.dense(p2_flat, 1024, activation=tf.nn.relu, use_bias=True, name="fc_1")

# Optional dropout
keep_prob = tf.placeholder(tf.float32)  # probability that each element is kept
f1_drop = tf.nn.dropout(f1, keep_prob)

# Final readout layer, alternative: tf.layers.dense(...)
f2 = tf.layers.dense(f1_drop, 10, activation=None, use_bias=True,  name="fc_2")

# Training
# Loss function
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_flatten, logits=f2))
# Adam optimizer, default parameters learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(cross_entropy)

# 0-1 loss
correct_prediction = tf.equal(tf.argmax(f2,1), y_flatten)  # second argmax argument specifies axis
# Average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initialize variables
    sess.graph.finalize()  # graph is read-only after this statement
    sess.run(ds_init_op)
    for i in tqdm(range(FLAGS.steps)):  # if you do not use tqdm,  write "... in range(FLAGS.steps):"
        try:
            x_train, y_train = sess.run(ds_next_element)
            train_step.run(feed_dict={x: x_train, y: y_train, keep_prob: 0.5})
        except tf.errors.OutOfRangeError:
            break

    print("test accuracy %g"%accuracy.eval(feed_dict={x: data_test_sub[0], y: data_test_sub[1], keep_prob: 1.0}))

