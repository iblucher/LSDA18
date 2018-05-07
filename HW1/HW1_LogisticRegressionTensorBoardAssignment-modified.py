# Logistic Regression
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/logreg', 'directory to put the summary data')
flags.DEFINE_string('data_dir', '../data', 'directory with data')
flags.DEFINE_integer('maxIter', 3000, 'number of iterations')
flags.DEFINE_float('learning_rate' , '0.1', 'learning rate of gradient descent')

# Read data
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTrain.csv',delimiter=',')
data_train = data[:,:-1]
data_train_label = data[:,-1].reshape(-1,1)
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTest.csv',delimiter=',')
data_test = data[:,:-1]
data_test_label = data[:,-1].reshape(-1,1)

print(data_train.shape)
print(data_test.shape)

# Input dimension
in_dim = data_train.shape[1]

# Initialize placeholders
x = tf.placeholder(shape=[None, in_dim], dtype=tf.float32, name='input')
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')

# TensorBoard will collapse the following nodes
with tf.name_scope('model') as scope:
    # Create variables for logistic regression
    A = tf.Variable(tf.random_normal(shape=[in_dim,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x, A), b)

print(FLAGS.learning_rate)

# Declare loss function
with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y))
    tf.summary.scalar('cross-entropy', loss)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
train_step = my_opt.minimize(loss)

# Map model output to binary predictions
with tf.name_scope('binary_prediction') as scope:
    prediction = tf.round(tf.sigmoid(model_output))
with tf.name_scope('0-1-loss') as scope:
    predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    # Logging
    merged = tf.summary.merge_all()  # collect all summaries in the graph
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement
    
    # Training loop
    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(FLAGS.maxIter):
        sess.run(train_step, feed_dict={x: data_train, y: data_train_label})
        summary = sess.run(merged, feed_dict={x: data_train, y: data_train_label})
        train_writer.add_summary(summary, i)
        summary = sess.run(merged, feed_dict={x: data_test, y: data_test_label})
        test_writer.add_summary(summary, i)

    writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

