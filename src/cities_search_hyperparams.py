# cities_search_hyperparams.py
# Daniel Lin, February 2019
# This script searches for hyperparameters to generate US city names

import rnn_lang_modeling as rnn
import load_cities_data as lcd
import numpy as np
import pandas as pd

TRAIN_SIZE = 0.6
NUM_EPOCHS = 6

def get_cost(rnn_model, seqs, vocab_size):
    """
    Calculates the cost of a recurrent neural network 
    
    Parameters:
        - rnn: the neural network
        - seqs: a list of lists representing the city names
        - vocab_size: the number of elements in the vocabulary
        
    Returns:
        - cost: the cost rnn gets on predicting the sequences
    """
    seqs_oh_X = rnn.get_one_hot(seqs, vocab_size, vocab_size-1, 
        shift_left=True)
    seqs_oh_Y = rnn.get_one_hot(seqs, vocab_size, vocab_size-1, 
        shift_left=False)        
    rnn_model.forward_pass(seqs_oh_X)
    cost = rnn_model.compute_cost(seqs_oh_Y, rnn_model.y_hats)
    return cost
    
def search_hyperparams(vocab_size, alpha_range, mbs_range, hid_units_range,
    max_grad_range, train_seqs, val_seqs, num_hp_samples=10):
    """
    Randomly tries different values for hyperparameters
    
    Parameters:
        - vocab_size: the number of elements in the vocabulary
        - alpha_range: a sequence with two elements, (a, b)
            the learning rate alpha will range over 10**a to 10**b
        - mbs_range: a sequence with two elements, (a, b)
            the mini-batch size will range from 2**a to 2**b
        - hid_units_range: a sequence with two elements, (a, b)
            the number of hidden units will range from a to b
        - max_grad_range: a sequence with two elements, (a, b)
            the max_grad will range from a to b
        - train_seqs: a list of lists, representing the training sequence data
        - val_seqs: a list of lists, representing the validation sequence data
        - num_hp_samples: the number of times to sample from the 
            space of hyperparameters
    
    Returns:
        - models_info: a list of tuples, where each tuple is:
            (training accuracy, validation accuracy, alpha, mini-batch size,
            number of hidden units, max_grad)
            The list is sorted descendingly by validation accuracy
    """
    models_info = []
    for i in range(num_hp_samples):
        print('Model %s out of %s' % ((i+1), num_hp_samples))
        # learning rate
        r = np.random.rand() * (alpha_range[1] - alpha_range[0]) \
            + alpha_range[0]
        alpha = 10 ** r
        
        # mini-batch size
        s = np.random.randint(mbs_range[0], mbs_range[1])
        mbs = 2 ** s
        
        # number of hidden units
        num_hidden_units = np.random.randint(hid_units_range[0], 
                                             hid_units_range[1])        
        
        # max_grad
        mg = np.random.randint(max_grad_range[0], max_grad_range[1])
        
        rnn_model = rnn.RNN(num_hidden_units, vocab_size, vocab_size)
        rnn_model.train_mini_batch(train_seqs, train_seqs, epochs=NUM_EPOCHS, 
            mini_batch_size=mbs, learning_rate=alpha, num_iter_msg=None, 
            max_grad=mg, shift_X=True)
        train_cost = get_cost(rnn_model, train_seqs, vocab_size)
        val_cost = get_cost(rnn_model, val_seqs, vocab_size)
        
        # add to list of models tried in format:
        # (training accuracy, validation accuracy, alpha, mini-batch size,
        # number of hidden units, max_grad)
        model_info = (train_cost, val_cost, alpha, mbs, num_hidden_units,
            mg)
        models_info.append(model_info)

    # print out the networks based on which had the best validation cost
    models_info.sort(reverse=False, key=lambda model: model[1])
    return models_info   
    
city_seqs, char_to_ind, ind_to_char, n_x = lcd.get_cities_data()
m = len(city_seqs)
train_num = int(m * TRAIN_SIZE)
seqs_train = city_seqs[:train_num]
seqs_val = city_seqs[train_num:]


# hyperparameter tuning
first_search = search_hyperparams(n_x, (-5, 0), (1, 10), (50, 200), (1, 5),
   seqs_train, seqs_val, num_hp_samples=10)
print(first_search)                       

