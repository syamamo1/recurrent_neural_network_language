import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from functools import reduce
from preprocess import get_data
import time


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.embedding_size = 128 #TODO
        self.batch_size = 128 #TODO

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))
        self.W = tf.Variable(tf.random.truncated_normal([2*self.embedding_size, self.vocab_size], stddev=.1))
        self.b = tf.Variable(tf.random.truncated_normal([self.vocab_size], stddev=.1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: probs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        #TODO: Fill in
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        concat = tf.concat([embedding[:,0], embedding[:,1]], 1)
        logits = tf.matmul(concat, self.W) + self.b
        probs = tf.nn.softmax(logits)

        return probs

    def loss_function(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: Please use np.reduce_mean and not np.reduce_sum when calculating your loss.
        
        :param probs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        #TODO: Fill in
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        av_loss = tf.reduce_mean(loss)

        return av_loss


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
    inds = tf.random.shuffle(range(len(train_labels)))
    shuffled_inputs = tf.gather(train_input, inds)
    shuffled_labels = tf.gather(train_labels, inds)

    num_iterations = int(len(shuffled_labels)/model.batch_size)
    for i in range(0, num_iterations):
        inputs = shuffled_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = shuffled_labels[i*model.batch_size:(i+1)*model.batch_size]
        
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = model.loss_function(predictions, labels)
            #print("Loss, {}% training steps complete: {}".format(round(100*i/num_iterations, 3), loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return None


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
    total_loss = 0
    num_iterations = int(len(test_labels)/model.batch_size)
    for i in range(0, num_iterations):
        inputs = test_input[i*model.batch_size:(i+1)*model.batch_size]
        labels = test_labels[i*model.batch_size:(i+1)*model.batch_size]

        predictions = model(inputs)
        total_loss += model.loss_function(predictions, labels)

    perplexity = np.exp(total_loss/num_iterations)
    return perplexity


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

    #print(" ".join(text))


def main():
    start_time = time.time()
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    test_dir = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\test.txt"
    train_dir = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\train.txt"
    train_data, test_data, vocabulary = get_data(train_dir, test_dir)    

    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = []
    for i in range(0, len(train_data)-2):
        train_inputs.append([train_data[i], train_data[i+1]])
    train_labels = train_data[2:]

    test_inputs = []
    for i in range(0, len(test_data)-2):
        test_inputs.append([test_data[i], test_data[i+1]])
    test_labels = test_data[2:]

    # TODO: initialize model
    mod = Model(len(vocabulary.keys()))

    # TODO: Set-up the training step
    train(mod, train_inputs, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(mod, test_inputs, test_labels)

    # Print out perplexity 
    print('Perplexity:', perplexity)
    
    # BONUS: Try printing out sentences with different starting words
    generate_sentence('famous', 'people', 5, vocabulary, mod)
    generate_sentence('famous', 'people', 10, vocabulary, mod)
    generate_sentence('i', 'like', 2, vocabulary, mod)
    generate_sentence('eat', 'my', 4, vocabulary, mod)
    generate_sentence('rhode', 'island', 5, vocabulary, mod)
    
    #print('Runtime:', (time.time()-start_time)/60)

    pass

if __name__ == '__main__':
    main()
