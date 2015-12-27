import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from sklearn import metrics


class RNNClassifier(object):
    def __init__(self,
                 vocabulary_size,
                 sequence_length,
                 batch_size,
                 num_classes,
                 embedding_size=128,
                 hidden_dim=256,
                 cell_class=rnn_cell.LSTMCell,
                 num_layers=1,
                 dropout_keep_prob_embedding=1.0,
                 dropout_keep_prob_affine=1.0,
                 dropout_keep_prob_cell_input=1.0,
                 dropout_keep_prob_cell_output=1.0):

        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_keep_prob_embedding = tf.constant(dropout_keep_prob_embedding)
        self.dropout_keep_prob_affine = tf.constant(dropout_keep_prob_affine)
        self.dropout_keep_prob_cell_input = tf.constant(dropout_keep_prob_cell_input)
        self.dropout_keep_prob_cell_output = tf.constant(dropout_keep_prob_cell_output)

    def build_graph(self, input_x, input_y):
        with tf.variable_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, input_x)
            self.embedded_chars_drop = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob_embedding)

        with tf.variable_scope("rnn") as scope:
            # The first RNN layer after the embedding
            first_cell = rnn_cell.DropoutWrapper(
                cell_class(self.hidden_dim, self.embedding_size),
                input_keep_prob=self.dropout_keep_prob_cell_input,
                output_keep_prob=self.dropout_keep_prob_cell_output)
            # The next RNN layer
            next_cell = rnn_cell.DropoutWrapper(
                cell_class(self.hidden_dim, self.hidden_dim),
                input_keep_prob=self.dropout_keep_prob_cell_input,
                output_keep_prob=self.dropout_keep_prob_cell_output)
            # The stacked layers
            self.cell = rnn_cell.MultiRNNCell([first_cell] + [next_cell] * (self.num_layers - 1))
            # Build the recurrence
            self.initial_state = tf.Variable(tf.zeros([self.batch_size, self.cell.state_size]))
            self.rnn_states = [self.initial_state]
            self.rnn_outputs = []
            for i in range(self.sequence_length):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = self.cell(self.embedded_chars_drop[:, i, :], self.rnn_states[-1])
                self.rnn_outputs.append(new_output)
                self.rnn_states.append(new_state)
            self.final_state = self.rnn_states[-1]
            self.final_output = self.rnn_outputs[-1]

        with tf.variable_scope("affine"):
            W = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.hidden_dim]), name="b")
            self.affine = tf.nn.tanh(tf.nn.xw_plus_b(self.final_output, W, b))
            self.affine_drop = tf.nn.dropout(self.affine, self.dropout_keep_prob_affine)

        with tf.variable_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.hidden_dim, self.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
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
    def __init__(self, clf, batch_size, num_epochs):
        self.clf = clf
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def eval(x_eval, y_eval):
        predictions = []
        true_labels = []
        total_loss = 0
        eval_batches = tfmodels.data.utils.batch_iter(
            list(zip(x_eval, y_eval)),
            batch_size=BATCH_SIZE,
            num_epochs=1)
        for batch in eval_batches:
            x_batch, y_batch = zip(*batch)
            if len(x_batch) < BATCH_SIZE:
                continue
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                self.clf.dropout_keep_prob_embedding: 1.0,
                self.clf.dropout_keep_prob_affine: 1.0,
                self.cfl.dropout_keep_prob_cell_input: 1.0,
                self.cfl.dropout_keep_prob_cell_output: 1.0
            }
            batch_predictions, batch_loss = sess.run([cnn.prediction, cnn.total_loss], feed_dict=feed_dict)
            true_labels = np.concatenate([true_labels, np.argmax(y_batch, axis=1)])
            predictions = np.concatenate([predictions, batch_predictions])
            total_loss += batch_loss
        acc = metrics.accuracy_score(true_labels, predictions)
        mean_loss = total_loss/len(y)
        return [acc, mean_loss]

    def train(sess, x_train, y_train, x_test, y_test, eval_every):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(cnn.total_loss, global_step=global_step)
        batches = tfmodels.data.utils.batch_iter(
            list(zip(x_train, y_train)),
            self.batch_size,
            self.num_epochs)
        for batch in batches:
            if len(batch) < clf.batch_size:
                print("WARNING: Batch was smaller than fixed batch size. Skipping.")
                continue
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch
            }
            _, _loss, _acc = sess.run([train_op, clf.mean_loss, clf.acc], feed_dict=feed_dict)
            current_step = tf.train.global_step(sess, global_step)
            print(".", end="", flush=True)
            if current_step % eval_every == 0:
                print("")
                train_acc, train_loss = eval_dataset(x_train, y_train)
                print("{}: Train Accuracy: {:g}, Train Mean Loss: {:g}".format(current_step, train_acc, train_loss))
                test_acc, test_loss = eval_dataset(x_test, y_test)
