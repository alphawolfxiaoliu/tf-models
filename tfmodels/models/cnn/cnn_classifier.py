import time
import numpy as np
import tensorflow as tf
import tfmodels.data.utils
from tfmodels.models.model_saver import ModelSaver


class CNNClassifier(ModelSaver):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    PARAMS = [
        "dropout_keep_prob_embedding",
        "dropout_keep_prob_features",
        "embedding_dim",
        "filter_sizes",
        "use_highway",
        "num_classes",
        "num_filters",
        "sequence_length",
        "vocabulary_size"
    ]

    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 num_classes=2,
                 embedding_dim=128,
                 filter_sizes="3,4,5",
                 num_filters=100,
                 use_highway=False,
                 dropout_keep_prob_embedding=1.0,
                 dropout_keep_prob_features=1.0):

        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.use_highway = use_highway
        self.dropout_keep_prob_embedding = dropout_keep_prob_embedding
        self.dropout_keep_prob_features = dropout_keep_prob_features

        # Parse filter sizes
        self.filter_sizes_arr = list(map(int, self.filter_sizes.split(",")))

    @staticmethod
    def add_flags():
        tf.flags.DEFINE_float("dropout_keep_prob_features", 1.0, "Output feature layer dropout")
        tf.flags.DEFINE_float("dropout_keep_prob_embedding", 1.0, "Embedding dropout")
        tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of embedding layer")
        tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
        tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated list of filter widths")
        tf.flags.DEFINE_boolean("use_highway", False, "Add highway layer on top of pooled features")

    def build_graph(self, input_x, input_y):

        # Placeholders for input, output and dropout
        self.input_x = input_x
        self.input_y = input_y

        self.dropout_keep_prob_embedding_t = tf.constant(self.dropout_keep_prob_embedding)
        self.dropout_keep_prob_features_t = tf.constant(self.dropout_keep_prob_features)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_drop = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob_embedding_t)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes_arr):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes_arr)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.features = self.h_pool_flat

        # Highway Layer
        if self.use_highway:
            with tf.name_scope("highway"):
                x = self.features
                W_h = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W_h")
                b_h = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b_h")
                W_t = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W_t")
                b_t = tf.Variable(tf.constant(-3.0, shape=[num_filters_total]), name="b_t")
                T = tf.nn.sigmoid(tf.nn.xw_plus_b(x, W_t, b_t), name="T")
                self.features = T * tf.nn.xw_plus_b(x, W_h, b_h) + (1.0 - T) * x

        # Add dropout
        with tf.name_scope("dropout"):
            self.features_dropped = tf.nn.dropout(self.features, self.dropout_keep_prob_features_t)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.features_dropped, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")


class CNNClassifierEvaluator:
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
        inc_example_count = self.example_count.assign_add(tf.cast(tf.shape(self.model.input_x)[0], "float"))

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
                self.model.dropout_keep_prob_features_t: 1.0
            }
            sess.run(self.eval_step, feed_dict=feed_dict)
        loss, acc, summaries, current_step = sess.run([self.mean_loss, self.accuracy, self.summaries, global_step])
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, current_step)
        return [loss, acc, current_step]


class CNNClassifierTrainer:
    """
    Helper class to train the CNNClassifier on batched data.
    """
    def __init__(self, model, optimizer=None, train_summary_dir=None, sess=None):
        sess = sess or tf.get_default_session()
        self.model = model
        with tf.variable_scope("training"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = optimizer or tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(model.total_loss, global_step=self.global_step)

            # Add summaries
            summary_mean_loss = tf.scalar_summary("mean_loss", model.mean_loss)
            summary_acc = tf.scalar_summary("accuracy", model.accuracy)
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
                [self.train_op, self.model.mean_loss, self.model.accuracy, self.global_step, self.train_summaries],
                feed_dict=feed_dict)
            if self.train_summary_writer is not None:
                self.train_summary_writer.add_summary(summaries, current_step)
            end_ts = time.time()
            yield train_loss, train_acc, current_step, (end_ts - start_ts)
