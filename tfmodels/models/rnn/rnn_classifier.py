import numpy as np
import tensorflow as tf
import tfmodels.data.utils
from tensorflow.models.rnn import rnn_cell
from sklearn import metrics


class RNNClassifier(object):
    def __init__(self,
                 embedding_dim=256,
                 hidden_dim=256,
                 affine_dim=256,
                 cell_class=rnn_cell.LSTMCell,
                 num_layers=2,
                 dropout_keep_prob_embedding=1.0,
                 dropout_keep_prob_affine=1.0,
                 dropout_keep_prob_cell_input=1.0,
                 dropout_keep_prob_cell_output=1.0):

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.affine_dim = affine_dim
        self.cell_class = cell_class
        self.num_layers = num_layers
        self.dropout_keep_prob_embedding = tf.constant(dropout_keep_prob_embedding)
        self.dropout_keep_prob_affine = tf.constant(dropout_keep_prob_affine)
        self.dropout_keep_prob_cell_input = tf.constant(dropout_keep_prob_cell_input)
        self.dropout_keep_prob_cell_output = tf.constant(dropout_keep_prob_cell_output)

    @staticmethod
    def add_flags():
        tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of embedding layer")
        tf.flags.DEFINE_integer("hidden_dim", 256, "Dimensionality of the RNN cells")
        tf.flags.DEFINE_integer("affine_dim", 256, "Dimensionality of affine layer at the last step")
        tf.flags.DEFINE_string("cell_class", "LSTM", "LSTM, GRU or BasicRNN")
        tf.flags.DEFINE_integer("num_layers", 2, "Number of stacked RNN cells")
        tf.flags.DEFINE_float("dropout_keep_prob_embedding", 1.0, "Embedding dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_affine", 1.0, "Affine output layer dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_input", 1.0, "RNN cell input connection dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_output", 1.0, "RNN cell output connection dropout")

    @staticmethod
    def from_flags():
        FLAGS = tf.flags.FLAGS
        cell_classes = {
            "LSTM": rnn_cell.LSTMCell,
            "GRU": rnn_cell.GRUCell,
            "BasicRNN": rnn_cell.BasicRNNCell,
        }
        return RNNClassifier(
            embedding_dim=FLAGS.embedding_dim,
            hidden_dim=FLAGS.hidden_dim,
            affine_dim=FLAGS.affine_dim,
            cell_class=cell_classes.get(FLAGS.cell_class, rnn_cell.LSTMCell),
            num_layers=FLAGS.num_layers,
            dropout_keep_prob_embedding=FLAGS.dropout_keep_prob_embedding,
            dropout_keep_prob_affine=FLAGS.dropout_keep_prob_affine,
            dropout_keep_prob_cell_input=FLAGS.dropout_keep_prob_cell_input,
            dropout_keep_prob_cell_output=FLAGS.dropout_keep_prob_cell_output
        )

    def build_graph(self, input_x, input_y, vocabulary_size):

        # Infer graph shapes from input tensor
        batch_size = input_x.get_shape().as_list()[0]
        sequence_length = input_x.get_shape().as_list()[1]
        num_classes = input_y.get_shape().as_list()[1]

        # Inputs
        self.input_x = input_x
        self.input_y = input_y

        with tf.variable_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocabulary_size, self.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, input_x)
            self.embedded_chars_drop = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob_embedding)

        with tf.variable_scope("rnn") as scope:
            # The first RNN layer after the embedding
            first_cell = rnn_cell.DropoutWrapper(
                self.cell_class(self.hidden_dim, self.embedding_dim),
                input_keep_prob=self.dropout_keep_prob_cell_input,
                output_keep_prob=self.dropout_keep_prob_cell_output)
            # The next RNN layer
            next_cell = rnn_cell.DropoutWrapper(
                self.cell_class(self.hidden_dim, self.hidden_dim),
                input_keep_prob=self.dropout_keep_prob_cell_input,
                output_keep_prob=self.dropout_keep_prob_cell_output)
            # The stacked layers
            self.cell = rnn_cell.MultiRNNCell([first_cell] + [next_cell] * (self.num_layers - 1))
            # Build the recurrence
            self.initial_state = tf.Variable(tf.zeros([batch_size, self.cell.state_size]))
            self.rnn_states = [self.initial_state]
            self.rnn_outputs = []
            for i in range(sequence_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(self.embedded_chars_drop[:, i, :], self.rnn_states[-1])
                self.rnn_outputs.append(new_output)
                self.rnn_states.append(new_state)
            self.final_state = self.rnn_states[-1]
            self.final_output = self.rnn_outputs[-1]

        with tf.variable_scope("affine"):
            W = tf.Variable(tf.truncated_normal([self.hidden_dim, self.affine_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.affine_dim]), name="b")
            self.affine = tf.nn.tanh(tf.nn.xw_plus_b(self.final_output, W, b))
            self.affine_drop = tf.nn.dropout(self.affine, self.dropout_keep_prob_affine)

        with tf.variable_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.affine_dim, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.affine_drop, W, b)
            self.predictions = tf.argmax(self.scores, 1)

        with tf.variable_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, input_y, name="ce_losses")
            self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)

        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class Trainer(object):
    def __init__(self, sess, clf, batch_size, num_epochs, eval_dev_every=None):
        self.sess = sess
        self.clf = clf
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_dev_every = eval_dev_every
        with sess.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.clf.total_loss, global_step=self.global_step)

    def eval(self, x_eval, y_eval):
        predictions = []
        labels = []
        total_loss = 0
        batches = tfmodels.data.utils.batch_iter(list(zip(x_eval, y_eval)), self.batch_size, self.num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            if len(x_batch) < self.batch_size:
                continue
            feed_dict = {
                self.clf.input_x: x_batch,
                self.clf.input_y: y_batch,
                self.clf.dropout_keep_prob_embedding: 1.0,
                self.clf.dropout_keep_prob_affine: 1.0,
                self.clf.dropout_keep_prob_cell_input: 1.0,
                self.clf.dropout_keep_prob_cell_output: 1.0
            }
            batch_predictions, batch_loss = self.sess.run(
                [self.clf.predictions, self.clf.total_loss],
                feed_dict=feed_dict)
            labels = np.concatenate([labels, np.argmax(y_batch, axis=1)])
            predictions = np.concatenate([predictions, batch_predictions])
            total_loss += batch_loss
        acc = metrics.accuracy_score(labels, predictions)
        mean_loss = total_loss/len(y_eval)
        return [acc, mean_loss]

    def train_iter(self, x_train, y_train):
        # Generate batches
        batches = tfmodels.data.utils.batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
        # Trainning loop
        for batch in batches:
            if len(batch) < self.batch_size:
                print("WARNING: Batch was too small, skipping: ({}) < ({})".format(len(batch), self.batch_size))
                continue
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                self.clf.input_x: x_batch,
                self.clf.input_y: y_batch
            }
            _, train_loss, train_acc, current_step = self.sess.run(
                [self.train_op, self.clf.mean_loss, self.clf.acc, self.global_step],
                feed_dict=feed_dict)
            yield [current_step, train_loss, train_acc]
