# MLP
#
# This example:
# - Trains a neural network. The Adam optimizer is employed. Its parameters may need tuning.
# - Monitors the learning using a validation data set.
# - Stores the network with the lowest validation error in a file. At the end, this network is loaded and evaluated.
# - Uses mini-batches and shuffles the data.


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/MLPMiniLog', 'directory to put the summary data')
flags.DEFINE_string('data_dir', '../data', 'directory with data')
flags.DEFINE_integer('max_iter', 7500, 'number of iterations')
flags.DEFINE_integer('max_epochs', 1000, 'number of epochs')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('no_hidden1', 64, 'size of first hidden layer')
flags.DEFINE_integer('no_hidden2', 64, 'size of second hidden layer')
flags.DEFINE_float('lr', 100, 'initial learning rate')
flags.DEFINE_integer('prefetch', 1, 'prefetch buffer size')

# Read and organize training data
in_dim = 13 # input dimension
record_defaults = [[1.0] for _ in range(in_dim)] # define all input features to be floats
record_defaults.append([1]) # add the label as an integer
def _parse_line(line):  # read a single line
    content = tf.decode_csv(line, record_defaults)  # decode line
    return tf.stack(content[:-1]), content[-1]  # return input and label

with tf.name_scope('training_data') as scope:
    ds = tf.data.TextLineDataset(FLAGS.data_dir + '/LSDAWeedCropTrain.csv')  # file to read line by line
    # Organize the data in randomized mini-batches of size batch_size,
    #   allow max_epochs passes through the whole data,
    #   set the size of the buffer used for shuffling the data
    ds = ds.apply(tf.contrib.data.shuffle_and_repeat(10*FLAGS.batch_size, FLAGS.max_epochs))
    ds = ds.apply(tf.contrib.data.map_and_batch(map_func=_parse_line, batch_size=FLAGS.batch_size, num_parallel_batches=4))
    ds = ds.prefetch(FLAGS.prefetch)

    # Create TensorFlow Iterator object
    ds_iterator = tf.data.Iterator.from_structure(ds.output_types,
                                              ds.output_shapes)
    ds_next_element = ds_iterator.get_next()
    ds_init_op = ds_iterator.make_initializer(ds)

# Read data sets, very inefficient to keep all data in main memory, but here the sets are small
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTrain.csv',delimiter=',')
data_train = data[:,:-1]
data_train_label = data[:,-1]
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropTest.csv',delimiter=',')
data_test = data[:,:-1]
data_test_label = data[:,-1]
data = np.genfromtxt(FLAGS.data_dir + '/LSDAWeedCropValidation.csv',delimiter=',')
data_validate = data[:,:-1]
data_validate_label = data[:,-1]

# Number of training data points
no_train = data_train.data.shape[0]
print("Numer of training data points:", no_train)

# Initialize placeholders
x = tf.placeholder(shape=[None, in_dim], dtype=tf.float32, name='input')
y = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

# Define model
y_1 = tf.layers.dense(x, FLAGS.no_hidden1, bias_initializer=tf.truncated_normal_initializer(stddev=0.1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=tf.sigmoid, name='layer_1')

# ?

model_output = tf.layers.dense(y_2, 1, bias_initializer=tf.truncated_normal_initializer(stddev=0.1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), activation=None, name='final_layer')
model_output = tf.squeeze(model_output)

# Declare loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y), name='mean_cross-entropy')
tf.summary.scalar('cross-entropy', loss)

# Declare optimizer
# ?
train_step = my_opt.minimize(loss)

# Map model output to binary predictions
with tf.name_scope('binary_predictions') as scope:
    prediction = tf.round(tf.sigmoid(model_output))
    predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct, name='mean_accuracy')
tf.summary.scalar('accuracy', accuracy)

# Logging
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
validate_writer=  tf.summary.FileWriter(FLAGS.summary_dir + '/validate')
saver = tf.train.Saver() # for storing the best network

with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement
    writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
    
    # Best validation accuracy seen so far
    best_validation = 0.0
    i = 0  # counter
    sess.run(ds_init_op)  # initialize iterator
    while True:
        try:
            x_train, y_train = sess.run(ds_next_element)
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
            i+=1
            summary = sess.run(merged, feed_dict={x: x_train, y: y_train})
            train_writer.add_summary(summary, i)
            if((i)%100==0):
                print("Iteration:", i, "/", FLAGS.max_iter)
                summary = sess.run(merged, feed_dict={x: data_test, y: data_test_label})
                test_writer.add_summary(summary, i)
                current_validation, summary = sess.run([accuracy, merged], feed_dict={x: data_validate, y: data_validate_label})
                validate_writer.add_summary(summary, i)
                if(current_validation > best_validation):
                    best_validation = current_validation
                    saver.save(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
                    print("\tbetter network stored,", current_validation, ">", best_validation)
            if(i==FLAGS.max_iter):
                break
        except tf.errors.OutOfRangeError:
            break

# Print values after last training step
    print("final training accuracy:", sess.run(accuracy, feed_dict={x: data_train, y: data_train_label}),
          "final test accuracy: ", sess.run(accuracy, feed_dict={x: data_test.data, y: data_test_label}),
          "final validation accuracy: ", sess.run(accuracy, feed_dict={x: data_validate, y: data_validate_label}))

# Load the network with the lowest validation error
with tf.Session() as sess:
    saver.restore(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
    print("best training accuracy:", sess.run(accuracy, feed_dict={x: data_train, y: data_train_label}),
          "best test accuracy: ", sess.run(accuracy, feed_dict={x: data_test.data, y: data_test_label}),
          "best validation accuracy: ", sess.run(accuracy, feed_dict={x: data_validate, y: data_validate_label}))



