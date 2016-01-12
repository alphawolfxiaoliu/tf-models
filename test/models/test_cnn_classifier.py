import unittest
import itertools
import tensorflow as tf
import numpy as np
import tfmodels.data.utils
import tempfile
from tfmodels.models.cnn.cnn_classifier import CNNClassifier, CNNClassifierTrainer, CNNClassifierEvaluator


BATCH_SIZE = 5
VOCABULARY_SIZE = 100
SEQUENCE_LENGTH = 8


class TestCNNClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def tearDown(self):
        self.sess.close()

    def _build_classifier(self):
        with self.graph.as_default(), self.sess.as_default():
            tf.set_random_seed(42)
            x = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH])
            y = tf.placeholder(tf.float32, [None, 2])
            cnn = CNNClassifier(
                sequence_length=SEQUENCE_LENGTH,
                vocabulary_size=VOCABULARY_SIZE,
                num_classes=2,
                embedding_dim=64,
                filter_sizes="2,3,4",
                num_filters=10,
                use_highway=False,
                dropout_keep_prob_embedding=1.0,
                dropout_keep_prob_features=1.0)
            cnn.build_graph(x, y)
            return cnn

    def test_prediction(self):
        with self.graph.as_default(), self.sess.as_default():
            cnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            feed_dict = {
                cnn.input_x: x_batch
            }
            predictions = self.sess.run(cnn.predictions, feed_dict)
            self.assertEqual(predictions.shape, (5,))

    def test_loss(self):
        with self.graph.as_default(), self.sess.as_default():
            cnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
            y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch
            }
            loss = self.sess.run(cnn.total_loss, feed_dict)
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
            cnn = self._build_classifier()
            t = CNNClassifierTrainer(cnn)
            self.sess.run(tf.initialize_all_variables())
            train_iter = t.train_loop(train_data_iter)
            for loss, acc, current_step, time_delta in train_iter:
                losses = losses + [loss]
        self.assertEqual(len(losses), 5)
        self.assertEqual(current_step, 5)

    def test_eval(self):
        x_dev = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE * 5, SEQUENCE_LENGTH])
        y_dev = np.eye(2)[np.random.randint(0, 1, [BATCH_SIZE * 5])]

        def make_eval_iter():
            data_iter = tfmodels.data.utils.batch_iter(
                list(zip(x_dev, y_dev)), BATCH_SIZE, 1, fill=True, seed=42)
            return map(lambda batch: zip(*batch), data_iter)

        with self.graph.as_default(), self.sess.as_default():
            cnn = self._build_classifier()
            ev = CNNClassifierEvaluator(cnn)
            self.sess.run(tf.initialize_all_variables())
            loss, acc, current_step = ev.eval(make_eval_iter())
            loss2, acc2, current_step = ev.eval(make_eval_iter())

        self.assertGreater(loss, 0)
        self.assertGreater(acc, 0)
        self.assertEqual(loss, loss2)
        self.assertEqual(acc, acc2)

    def test_save_load(self):
        # Data
        x_batch = np.random.randint(0, VOCABULARY_SIZE, [BATCH_SIZE, SEQUENCE_LENGTH])
        y_batch = np.eye(2)[np.ones(BATCH_SIZE, dtype=np.int32)]

        _, checkpoint_path = tempfile.mkstemp()

        with self.graph.as_default(), self.sess.as_default():
            cnn = self._build_classifier()
            self.sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch}
            loss1 = self.sess.run(cnn.total_loss, feed_dict)
            saver.save(self.sess, checkpoint_path)

        # Save model hyperparameters
        f, path = tempfile.mkstemp()
        cnn.save_to_file(path)

        with tf.Graph().as_default(), tf.Session() as sess:
            cnn2 = CNNClassifier.load_from_file(path)
            x = tf.placeholder(tf.int32, [BATCH_SIZE, SEQUENCE_LENGTH])
            y = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
            cnn2.build_graph(x, y)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            feed_dict = {cnn2.input_x: x_batch, cnn2.input_y: y_batch}
            loss2 = sess.run(cnn2.total_loss, feed_dict)

        self.assertEqual(loss1, loss2)
