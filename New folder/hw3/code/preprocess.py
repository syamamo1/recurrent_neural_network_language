import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # TODO: load and concatenate training data from training file.
    with open(train_file) as train_set:
        a = np.array(train_set.read().split())
        # print('train set', a, len(a))

    # TODO: load and concatenate testing data from testing file.
    with open(test_file) as test_set:
        b = np.array(test_set.read().split())
        # print('test set:', b, len(b))

    # TODO: read in and tokenize training data
    unique_id = 0
    word2id = {}
    train_words = []
    for word in a:
        if word not in word2id.keys():
            word2id[word] = unique_id
            unique_id += 1
        train_words.append(word2id[word])

    # TODO: read in and tokenize testing data
    test_words = []
    for word in b:
        test_words.append(word2id[word])

    # BONUS: Ensure that all words appearing in test also appear in train <-- No thanks

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    # print('dict len:', len(word2id))
    return train_words, test_words, word2id

# test_data = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\test.txt"
# train_data = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\train.txt"
# get_data(train_data, test_data)
