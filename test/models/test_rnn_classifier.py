import unittest
import itertools
import tensorflow as tf
import numpy as np
import tfmodels.data.utils
from tensorflow.models.rnn import rnn_cell
from tfmodels.models.rnn.rnn_classifier import RNNClassifier, RNNClassifierTrainer, RNNClassifierEvaluator


BATCH_SIZE = 5
VOCABULARY_SIZE = 100
SEQUENCE_LENGTH = 8


class TestRNNClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def tearDown(self):
        self.sess.close()

    def _build_classifier(self):
        with self.graph.as_default(), self.sess.as_default():
            tf.set_random_seed(42)
            x = tf.placeholder(tf.int32, [BATCH_SIZE, SEQUENCE_LENGTH])
            y = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
            rnn = RNNClassifier(
                sequence_length=SEQUENCE_LENGTH,
                vocabulary_size=VOCABULARY_SIZE,
                num_classes=2,
                batch_size=BATCH_SIZE,
                backprop_truncate_after=SEQUENCE_LENGTH,
                embedding_dim=128,
                hidden_dim=128,
                affine_dim=128,
                cell_class="LSTM",
                num_layers=2,
                dropout_keep_prob_embedding=0.3,
                dropout_keep_prob_affine=0.5,
                dropout_keep_prob_cell_input=1.0,
                dropout_keep_prob_cell_output=1.0)
            rnn.build_graph(x, y)
            return rnn

    def test_prediction(self):
        with self.graph.as_default(), self.sess.as_default():
            rnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            feed_dict = {
                rnn.input_x: x_batch
            }
            predictions = self.sess.run(rnn.predictions, feed_dict)
            self.assertEqual(predictions.shape, (5,))

    def test_loss(self):
        with self.graph.as_default(), self.sess.as_default():
            rnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch
            }
            loss = self.sess.run(rnn.total_loss, feed_dict)
            self.assertGreater(loss, 0)

    def test_dropout_feed(self):
        with self.graph.as_default(), self.sess.as_default():
            rnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
                rnn.dropout_keep_prob_embedding: 1.0,
                rnn.dropout_keep_prob_affine: 1.0,
                rnn.dropout_keep_prob_cell_input: 1.0,
                rnn.dropout_keep_prob_cell_output: 1.0
            }
            loss = self.sess.run(rnn.total_loss, feed_dict)
            self.assertGreater(loss, 0)

    def test_trainer(self):
        num_batches = 5
        x_train = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE * num_batches, SEQUENCE_LENGTH])
        y_train = np.eye(2)[np.ones(BATCH_SIZE * num_batches, dtype=np.int32)]
        train_data_iter = tfmodels.data.utils.batch_iter(
            list(zip(x_train, y_train)), BATCH_SIZE, 1, fill=True, seed=42)
        train_data_iter = map(lambda batch: zip(*batch), train_data_iter)
        losses = []
        with self.graph.as_default(), self.sess.as_default():
            rnn = self._build_classifier()
            t = RNNClassifierTrainer(rnn)
            self.sess.run(tf.initialize_all_variables())
            train_iter = t.train_loop(train_data_iter)
            for loss, acc, current_step, time_delta in train_iter:
                losses = losses + [loss]
        self.assertEqual(len(losses), 5)
        self.assertEqual(current_step, 5)

    def test_eval(self):
        x_dev = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE * 5, SEQUENCE_LENGTH])
        y_dev = np.eye(2)[np.ones(BATCH_SIZE * 5, dtype=np.int32)]
        data_iter = tfmodels.data.utils.batch_iter(
            list(zip(x_dev, y_dev)), BATCH_SIZE, 1, fill=True, seed=42)
        data_iter = map(lambda batch: zip(*batch), data_iter)
        with self.graph.as_default(), self.sess.as_default():
            rnn = self._build_classifier()
            ev = RNNClassifierEvaluator(rnn)
            self.sess.run(tf.initialize_all_variables())
            loss, acc, current_step = ev.eval(data_iter)
        self.assertGreater(loss, 0)
        self.assertGreater(acc, 0)
