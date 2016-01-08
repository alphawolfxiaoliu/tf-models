import time
import numpy as np
import tensorflow as tf
import tfmodels.data.utils
from tfmodels.models.model_saver import ModelSaver
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn as tf_rnn


class RNNClassifier(ModelSaver):
    """
    Recurrent Neural Network Classifier that makes a prediction at the last time step.
    """

    PARAMS = [
        "sequence_length", "vocabulary_size", "num_classes", "batch_size", "backprop_truncate_after",
        "embedding_dim", "cell_class", "hidden_dim", "affine_dim", "num_layers",
        "dropout_keep_prob_embedding", "dropout_keep_prob_affine", "dropout_keep_prob_cell_input",
        "dropout_keep_prob_cell_output"]

    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 num_classes,
                 batch_size=64,
                 backprop_truncate_after=256,
                 embedding_dim=256,
                 hidden_dim=256,
                 affine_dim=256,
                 cell_class="LSTM",
                 num_layers=2,
                 dropout_keep_prob_embedding=1.0,
                 dropout_keep_prob_affine=1.0,
                 dropout_keep_prob_cell_input=1.0,
                 dropout_keep_prob_cell_output=1.0):

        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.backprop_truncate_after = backprop_truncate_after
        self.embedding_dim = embedding_dim
        self.cell_class = cell_class
        self.hidden_dim = hidden_dim
        self.affine_dim = affine_dim
        self.num_layers = num_layers
        self.dropout_keep_prob_embedding = dropout_keep_prob_embedding
        self.dropout_keep_prob_affine = dropout_keep_prob_affine
        self.dropout_keep_prob_cell_input = dropout_keep_prob_cell_input
        self.dropout_keep_prob_cell_output = dropout_keep_prob_cell_output
        self.cell_class_map = {
            "LSTM": rnn_cell.BasicLSTMCell,
            "GRU": rnn_cell.GRUCell,
            "BasicRNN": rnn_cell.BasicRNNCell,
        }

    @staticmethod
    def add_flags():
        tf.flags.DEFINE_integer("affine_dim", 256, "Dimensionality of affine layer at the last step")
        tf.flags.DEFINE_integer("batch_size", 64, "Size for one batch of training/dev examples")
        tf.flags.DEFINE_string("cell_class", "LSTM", "LSTM, GRU or BasicRNN")
        tf.flags.DEFINE_float("dropout_keep_prob_affine", 1.0, "Affine output layer dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_input", 1.0, "RNN cell input connection dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_cell_output", 1.0, "RNN cell output connection dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_embedding", 1.0, "Embedding dropout")
        tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of embedding layer")
        tf.flags.DEFINE_integer("hidden_dim", 256, "Dimensionality of the RNN cells")
        tf.flags.DEFINE_integer("num_layers", 2, "Number of stacked RNN cells")
        tf.flags.DEFINE_integer("backprop_truncate_after", 256, "Truncate backpropagation after this many steps")

    def build_graph(self, input_x, input_y):
        """
        Builds the graph. Call this after instantiating the class.
        """
        self.input_x = input_x
        self.input_y = input_y

        self.dropout_keep_prob_embedding_t = tf.constant(self.dropout_keep_prob_embedding)
        self.dropout_keep_prob_affine_t = tf.constant(self.dropout_keep_prob_affine)
        self.dropout_keep_prob_cell_input_t = tf.constant(self.dropout_keep_prob_cell_input)
        self.dropout_keep_prob_cell_output_t = tf.constant(self.dropout_keep_prob_cell_output)

        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            W = tf.get_variable(
                "W",
                [self.vocabulary_size, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(W, input_x)
            self.embedded_chars_drop = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob_embedding_t)

        with tf.variable_scope("rnn") as scope:
            # The RNN cell
            cell_class = self.cell_class_map.get(self.cell_class)
            one_cell = rnn_cell.DropoutWrapper(
                cell_class(self.hidden_dim),
                input_keep_prob=self.dropout_keep_prob_cell_input_t,
                output_keep_prob=self.dropout_keep_prob_cell_output_t)
            self.cell = rnn_cell.MultiRNNCell([one_cell] * self.num_layers)
            # Build the recurrence. We do this manually to use truncated backprop
            self.initial_state = tf.zeros([self.batch_size, self.cell.state_size])
            self.rnn_states = [self.initial_state]
            self.rnn_outputs = []
            for i in range(self.sequence_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(self.embedded_chars_drop[:, i, :], self.rnn_states[-1])
                if i < max(0, self.sequence_length - self.backprop_truncate_after):
                    new_state = tf.stop_gradient(new_state)
                self.rnn_outputs.append(new_output)
                self.rnn_states.append(new_state)
            self.final_state = self.rnn_states[-1]
            self.final_output = self.rnn_outputs[-1]

        with tf.variable_scope("affine"):
            W = tf.get_variable(
                "W",
                [self.hidden_dim, self.affine_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.affine_dim],
                initializer=tf.constant_initializer(0.1))
            self.affine = tf.nn.tanh(tf.nn.xw_plus_b(self.final_output, W, b))
            self.affine_drop = tf.nn.dropout(self.affine, self.dropout_keep_prob_affine_t)

        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                [self.affine_dim, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                "b",
                [self.num_classes],
                initializer=tf.constant_initializer(0.1))
            self.scores = tf.nn.xw_plus_b(self.affine_drop, W, b)
            self.predictions = tf.argmax(self.scores, 1)

        with tf.variable_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, input_y, name="ce_losses")
            self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)

        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")


class RNNClassifierEvaluator:
    """
    Helper class to evaluate a dataset on the classifier.
    Stores temp data in graph variables to sum up loss and accuracy over multiple batches of the data set.
    """
    def __init__(self, model, summary_dir=None):
        self.model = model
        with tf.variable_scope("evaluation"):
            self.summary_writer = None
            if summary_dir is not None:
                self.summary_writer = tf.train.SummaryWriter(summary_dir)
            self.build_eval_graph()

    def build_eval_graph(self):
        # Keep track of the totals while running through the batch data
        self.total_loss = tf.Variable(0.0, trainable=False, collections=[])
        self.total_correct = tf.Variable(0.0, trainable=False, collections=[])
        self.example_count = tf.Variable(0.0, trainable=False, collections=[])

        # Calculates the means
        self.mean_loss = self.total_loss / self.example_count
        self.accuracy = self.total_correct / self.example_count

        # Operations to modify to the stateful variables
        inc_total_loss = self.total_loss.assign_add(self.model.total_loss)
        inc_total_correct = self.total_correct.assign_add(
            tf.reduce_sum(tf.cast(self.model.correct_predictions, "float")))
        inc_example_count = self.example_count.assign_add(self.model.batch_size)

        # Operation to reset all the stateful vars. Should be called before starting a data set evaluation.
        with tf.control_dependencies(
                [self.total_loss.initializer, self.total_correct.initializer, self.example_count.initializer]):
            self.eval_reset = tf.no_op()

        # Operation to modify the stateful variables with data from one batch
        # Should be called for each batch in the evaluatin set
        with tf.control_dependencies([inc_total_loss, inc_total_correct, inc_example_count]):
            self.eval_step = tf.no_op()

        # Summaries
        summary_mean_loss = tf.scalar_summary("mean_loss", self.mean_loss)
        summary_acc = tf.scalar_summary("accuracy", self.accuracy)
        self.summaries = tf.merge_summary([summary_mean_loss, summary_acc])

    def eval(self, xy_iter, global_step=None, sess=None):
        sess = sess or tf.get_default_session()
        global_step = global_step or tf.no_op()
        sess.run(self.eval_reset)
        for x_batch, y_batch in xy_iter:
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.dropout_keep_prob_embedding_t: 1.0,
                self.model.dropout_keep_prob_affine_t: 1.0,
                self.model.dropout_keep_prob_cell_input_t: 1.0,
                self.model.dropout_keep_prob_cell_output_t: 1.0
            }
            sess.run(self.eval_step, feed_dict=feed_dict)
        loss, acc, summaries, current_step = sess.run([self.mean_loss, self.accuracy, self.summaries, global_step])
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, current_step)
        return [loss, acc, current_step]


class RNNClassifierTrainer:
    """
    Helper class to train the RNNClassifier on batched data.
    """
    def __init__(self, model, optimizer=None, train_summary_dir=None, sess=None, max_grad_norm=5):
        sess = sess or tf.get_default_session()
        self.model = model
        with tf.variable_scope("training"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            # Clip gradients and apply them
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.total_loss, tvars), max_grad_norm)
            self.optimizer = optimizer or tf.train.AdamOptimizer()
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            # Add summaries
            summary_mean_loss = tf.scalar_summary("mean_loss", model.mean_loss)
            summary_acc = tf.scalar_summary("accuracy", model.acc)
            self.train_summaries = tf.merge_summary([summary_mean_loss, summary_acc])
            self.train_summary_writer = None
            if train_summary_dir is not None:
                self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

    def train_loop(self, xy_iter, sess=None):
        sess = sess or tf.get_default_session()
        for x_batch, y_batch in xy_iter:
            start_ts = time.time()
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch
            }
            _, train_loss, train_acc, current_step, summaries = sess.run(
                [self.train_op, self.model.mean_loss, self.model.acc, self.global_step, self.train_summaries],
                feed_dict=feed_dict)
            if self.train_summary_writer is not None:
                self.train_summary_writer.add_summary(summaries, current_step)
            end_ts = time.time()
            yield train_loss, train_acc, current_step, (end_ts - start_ts)
