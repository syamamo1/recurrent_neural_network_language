import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
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
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 50 #TODO
        self.batch_size = 64 #TODO 

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))

        self.rnn_layer = tf.keras.layers.LSTM(self.vocab_size, return_sequences=True, return_state=True)
        self.linear_layer = tf.keras.layers.Dense(self.vocab_size, activation='relu')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        #TODO: Fill in 
        if initial_state is None:
            embedding = tf.nn.embedding_lookup(self.E, inputs) # embedding [batch_size, window_size, embedding_size] 
            whole_seq_output, final_memory_state, final_carry_state = self.rnn_layer(embedding) # whole_seq_output [batch_size, window_size, units]
            logits = self.linear_layer(whole_seq_output) # logits [batch_size, window_size, units=vocab_size]
            probs = tf.nn.softmax(logits) # probs [batch_size, vocab_size]
            return probs, (final_memory_state, final_carry_state)
            


    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        av_loss = tf.reduce_mean(loss)

        return av_loss


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    real_inputs = []
    real_labels = []
    for i in range(0, len(train_inputs)-model.window_size):
        real_inputs.append(train_inputs[i:i+model.window_size])
        real_labels.append(train_labels[i:i+model.window_size])

    inds = tf.random.shuffle(range(len(real_labels)))
    shuffled_inputs = tf.gather(real_inputs, inds)
    shuffled_labels = tf.gather(real_labels, inds)

    num_iterations = int(len(shuffled_labels)/model.batch_size)
    for i in range(0, num_iterations):
        inputs = shuffled_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = shuffled_labels[i*model.batch_size:(i+1)*model.batch_size]
        
        with tf.GradientTape() as tape:
            predictions, state = model.call(inputs, None)
            loss = model.loss(predictions, labels)
            #print("{}% training steps complete: Loss: {}".format(round(100*i/num_iterations, 3), round(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    real_inputs = []
    real_labels = []
    for i in range(0, len(test_inputs)-model.window_size):
        real_inputs.append(test_inputs[i:i+model.window_size])
        real_labels.append(test_labels[i:i+model.window_size])   

    total_loss = 0
    num_iterations = int(len(real_labels)/model.batch_size)
    for i in range(0, num_iterations):
        inputs = real_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = real_labels[i*model.batch_size:(i+1)*model.batch_size]

        predictions, state = model.call(inputs, None)
        total_loss += model.loss(predictions, labels)

    perplexity = np.exp(total_loss/num_iterations)
    return perplexity



def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    #print(" ".join(text))


def main():
    start_time = time.time()
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    test_dir = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\test.txt"
    train_dir = "C:\\Users\smy18\workspace2\dl\hw3-lm-syamamo1\hw3\data\\train.txt"
    train_data, test_data, vocabulary = get_data(train_dir, test_dir)   
    
    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs, train_labels = train_data[:-1], train_data[1:]
    test_inputs, test_labels = test_data[:-1], test_data[1:]

    # TODO: initialize model
    mod = Model(len(vocabulary.keys()))

    # TODO: Set-up the training step
    train(mod, train_inputs, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(mod, test_inputs, test_labels)

    # Print out perplexity 
    print('Perplexity:', perplexity)

    # BONUS: Try printing out various sentences with different start words and sample_n parameters 
    # generate_sentence('famous', 'people', 5, vocabulary, mod)
    # generate_sentence('i', 'like', 2, vocabulary, mod)
    # generate_sentence('eat', 'my', 4, vocabulary, mod)
    # generate_sentence('rhode', 'island', 5, vocabulary, mod)
    
    #print('Runtime:', (time.time()-start_time)/60)
    
    pass

if __name__ == '__main__':
    main()
