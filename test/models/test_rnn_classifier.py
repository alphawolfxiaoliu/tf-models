import unittest
import itertools
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tfmodels.models.rnn.rnn_classifier import RNNClassifier, Trainer

BATCH_SIZE = 5
VOCABULARY_SIZE = 100
SEQUENCE_LENGTH = 50


class TestRNNClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default(), self.sess.as_default():
            tf.set_random_seed(42)
            self.x = tf.placeholder(tf.int32, [BATCH_SIZE, SEQUENCE_LENGTH])
            self.y = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
            self.rnn = RNNClassifier(
                vocabulary_size=100,
                sequence_length=SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                num_classes=2,
                embedding_size=256,
                hidden_dim=256,
                cell_class=rnn_cell.LSTMCell,
                num_layers=2,
                dropout_keep_prob_embedding=0.3,
                dropout_keep_prob_affine=0.5,
                dropout_keep_prob_cell_input=1.0,
                dropout_keep_prob_cell_output=1.0)
            self.rnn.build_graph(self.x, self.y)
            self.sess.run(tf.initialize_all_variables())

    def teatDown(self):
        self.sess.close()

    def test_prediction(self):
        with self.sess.as_default():
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            feed_dict = {
                self.x: x_batch
            }
            predictions = self.sess.run(self.rnn.predictions, feed_dict)
            self.assertEqual(predictions.shape, (5,))

    def test_loss(self):
        with self.graph.as_default(), self.sess.as_default():
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch
            }
            loss = self.sess.run(self.rnn.total_loss, feed_dict)
            self.assertAlmostEqual(loss, 3.1754146)

    def test_dropout_feed(self):
        with self.graph.as_default(), self.sess.as_default():
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.rnn.dropout_keep_prob_embedding: 1.0,
                self.rnn.dropout_keep_prob_affine: 1.0,
                self.rnn.dropout_keep_prob_cell_input: 1.0,
                self.rnn.dropout_keep_prob_cell_output: 1.0
            }
            loss = self.sess.run(self.rnn.total_loss, feed_dict)
            self.assertAlmostEqual(loss, 3.1808105)

    def test_trainer_train(self):
        x_train = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE * 5, SEQUENCE_LENGTH])
        y_train = np.eye(2)[np.ones(BATCH_SIZE * 5, dtype=np.int32)]
        losses = []
        with self.graph.as_default(), self.sess.as_default():
            t = Trainer(self.sess, self.rnn, BATCH_SIZE, 2)
            self.sess.run(tf.initialize_all_variables())
            train_iter = t.train_iter(x_train, y_train, x_train, y_train)
            for current_step, loss, acc in train_iter:
                losses = losses + [loss]
        self.assertEqual(len(losses), 10)

    def test_trainer_eval(self):
        x_dev = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE * 5, SEQUENCE_LENGTH])
        y_dev = np.eye(2)[np.ones(BATCH_SIZE * 5, dtype=np.int32)]
        with self.graph.as_default(), self.sess.as_default():
            tf.set_random_seed(42)
            t = Trainer(self.sess, self.rnn, BATCH_SIZE, 2)
            self.sess.run(tf.initialize_all_variables())
            acc, mean_loss = t.eval(x_dev, y_dev)
        self.assertAlmostEqual(acc, 0.5600000)
        self.assertAlmostEqual(mean_loss, 1.369157199859619)
