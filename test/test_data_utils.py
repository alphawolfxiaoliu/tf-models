import unittest
import tfmodels.data.utils
import itertools


class TestDataUtils(unittest.TestCase):
    def test_build_vocabulary(self):
        test_sequence = [["Hi", "there", ",", "girl", "!"], ["Hello", "!"]]
        vocab, vocab_inverse = tfmodels.data.utils.build_vocabulary(test_sequence)
        self.assertEqual(len(vocab), 6)
        self.assertEqual(len(vocab_inverse), 6)
        for idx in range(len(vocab)):
            self.assertEqual(vocab[vocab_inverse[idx]], idx)

    def test_pad_sequences_left(self):
        sequences = [["Hello", "there"], ["This", "works", "!"], ["No"]]
        result = tfmodels.data.utils.pad_sequences(sequences, pad_location="LEFT")
        self.assertEqual(result, [["[PAD]", "Hello", "there"], ["This", "works", "!"], ["[PAD]", "[PAD]", "No"]])

    def test_pad_sequences_right(self):
        sequences = [["Hello", "there"], ["This", "works", "!"], ["No"]]
        result = tfmodels.data.utils.pad_sequences(sequences, pad_location="RIGHT")
        self.assertEqual(result, [["Hello", "there", "[PAD]"], ["This", "works", "!"], ["No", "[PAD]", "[PAD]"]])

    def test_batch_iter(self):
        data = [1, 2, 3]
        batches = list(tfmodels.data.utils.batch_iter(data, batch_size=2, num_epochs=3))
        self.assertEqual(len(batches), 6)
        flattened = list(itertools.chain(*batches))
        self.assertEqual(len(flattened), 9)
        for item in data:
            self.assertEqual(sum(1 for _ in flattened if _ == item), 3)
