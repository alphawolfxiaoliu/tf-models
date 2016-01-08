import numpy as np
import tensorflow as tf
import tfmodels.data.utils
from tfmodels.data.mr import MRData
from tfmodels.data.imdb import IMDBData
from datetime import datetime
import time
import os
from tfmodels.models.rnn.rnn_classifier import RNNClassifier, RNNClassifierTrainer, RNNClassifierEvaluator

# Pick training data
tf.flags.DEFINE_boolean("data_mr", False, "Dataset: MR")
tf.flags.DEFINE_boolean("data_sst", False, "Dataset: SST")

# Classifier parameters
RNNClassifier.add_flags()

# Training parameters
tf.flags.DEFINE_integer("random_state", 42, "Random state initialization for reproducibility")
tf.flags.DEFINE_integer("max_sequence_length", 128, "Examples will be padded/truncated to this length")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this number of steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Evaluate model on dev set after this number of steps")

# Session Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow soft device placement (e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Print parameters
FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

np.random.seed(FLAGS.random_state)

# Load the Training data
if FLAGS.data_mr:
    print("Loading MR data...", flush=True)
    data = MRData()
elif FLAGS.data_sst:
    print("Loading SST data...", flush=True)
    data = IMDBData(padding=True, clean_str=True, max_length=FLAGS.max_sequence_length)

vocab, vocab_inv = data.vocab, data.vocab_inv
x_train, x_dev, y_train, y_dev = data.build_train_dev()

# Parameters
SEQUENCE_LENGTH = x_train.shape[1]
VOCABULARY_SIZE = len(vocab)
train_data_iter = tfmodels.data.utils.xy_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)

# Create a graph and session
graph = tf.Graph()
session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(graph=graph, config=session_conf)

with graph.as_default(), sess.as_default():
    tf.set_random_seed(FLAGS.random_state)

    # Build the model
    model_params = {"sequence_length": SEQUENCE_LENGTH, "vocabulary_size": VOCABULARY_SIZE, "num_classes": 2}
    model_params.update(FLAGS.__flags)
    model = RNNClassifier.from_dict(model_params)
    x = tf.placeholder(tf.int32, [FLAGS.batch_size, SEQUENCE_LENGTH])
    y = tf.placeholder(tf.float32, [FLAGS.batch_size, 2])
    model.build_graph(x, y)

    # Directory for training and dev summaries
    timestamp = str(int(time.time()))
    rundir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    train_dir = os.path.join(rundir, "train")
    dev_dir = os.path.join(rundir, "dev")

    # Build the Trainer/Evaluator
    trainer = RNNClassifierTrainer(model, train_summary_dir=train_dir)
    evaluator = RNNClassifierEvaluator(model, summary_dir=dev_dir)

    # Saving/Checkpointing
    checkpoint_dir = os.path.join(rundir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    # Initialization, optinally load from checkpoint
    sess.run(tf.initialize_all_variables())
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Restoring checkpoint from {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Initialize variables
    sess.run(tf.initialize_all_variables())

    # Training loop
    for train_loss, train_acc, current_step, time_delta in trainer.train_loop(train_data_iter):
        examples_per_second = FLAGS.batch_size/time_delta
        print("{}: step {}, loss {:g}, acc {:g} ({:g} examples/sec)".format(
                datetime.now().isoformat(), current_step, train_loss, train_acc, examples_per_second))

        # Evaluate dev set
        if current_step % FLAGS.evaluate_every == 0:
            dev_iter = tfmodels.data.utils.xy_iter(x_dev, y_dev, FLAGS.batch_size, 1)
            mean_loss, acc, _ = evaluator.eval(dev_iter, global_step=trainer.global_step)
            print("{}: Step {}, Dev Accuracy: {:g}, Dev Mean Loss: {:g}".format(
                datetime.now().isoformat(), current_step, acc, mean_loss))

        # Checkpoint Model
        if current_step % FLAGS.checkpoint_every == 0:
            save_path = saver.save(sess, checkpoint_file, global_step=trainer.global_step)
            print("Saved {}".format(save_path))
