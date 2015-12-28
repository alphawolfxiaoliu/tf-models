import numpy as np
import tensorflow as tf
import tfmodels.data.utils
import tfmodels.data.mr
from tfmodels.models.rnn.rnn_classifier import RNNClassifier, Trainer

# Classifier parameters
RNNClassifier.add_flags()
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Training batch size")
tf.flags.DEFINE_integer("max_sequence_length", 128, "Maximum sequence length")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 16, "Evaluate model on dev set after this number of steps")
# Session Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement (e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Print parameters
FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preprocessing
sentences, labels = tfmodels.data.mr.load()
split_sentences = [s.split(" ") for s in sentences]
# Sequence length: min(longest sentences, sequence length flag)
sequence_length = min(np.max([len(s) for s in split_sentences]), FLAGS.max_sequence_length)
padded_sentences = tfmodels.data.utils.pad_sequences(split_sentences, pad_location="LEFT")
vocab, vocab_inverse = tfmodels.data.utils.build_vocabulary(padded_sentences)

x_train = [[vocab[token] for token in seq] for seq in padded_sentences]
y_train = np.eye(2)[labels]


# Create a graph and session
graph = tf.Graph()
session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(graph=graph, config=session_conf)

with graph.as_default(), sess.as_default():
    # Build the model
    x = tf.placeholder(tf.int32, [FLAGS.batch_size, sequence_length])
    y = tf.placeholder(tf.float32, [FLAGS.batch_size, 2])
    rnn = RNNClassifier.from_flags()
    rnn.build_graph(x, y, len(vocab))

    # Build the trainer
    t = Trainer(sess, rnn, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    # Initialize variables
    sess.run(tf.initialize_all_variables())

    # Train oop
    train_iter = t.train_iter(x_train, y_train, x_train, y_train)
    for current_step, train_loss, train_acc in train_iter:
        print("{}: Train Accuracy: {:g}, Train Mean Loss: {:g}".format(current_step, train_acc, train_loss))
