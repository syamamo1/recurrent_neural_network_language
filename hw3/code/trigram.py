import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from functools import reduce
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.embedding_size = _ #TODO
        self.batch_size = _ #TODO

        # TODO: initialize embeddings and forward pass weights (weights, biases)

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: probs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        #TODO: Fill in

        pass

    def loss_function(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: Please use np.reduce_mean and not np.reduce_sum when calculating your loss.
        
        :param probs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        #TODO: Fill in

        pass


def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    
    #TODO Fill in

    pass


def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)

    pass  


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    #NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(
            model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess

    # TO-DO:  Separate your train and test data into inputs and labels

    # TODO: initialize model

    # TODO: Set-up the training step

    # TODO: Set up the testing steps

    # Print out perplexity 
    
    # BONUS: Try printing out sentences with different starting words  
    
    pass

if __name__ == '__main__':
    main()
