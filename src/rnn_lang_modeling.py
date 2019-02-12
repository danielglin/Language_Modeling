# rnn_lang_modeling.py
# Daniel Lin, February 2019
# This script implements a vanilla RNN for language modeling

import numpy as np

class RNN:
    """
    Builds a recurrent neural network that uses a tanh activation function
    Is a many-many architecture
    
    Parameters:
        - hidden_state_size: the number of units in the hidden state
        - input_vocab_size: the size of the input vocabulary
        - output_vocab_size: the size of the output vocabulary
    """
    def __init__(self, hidden_state_size, input_vocab_size, output_vocab_size):

        self.n_a = hidden_state_size
        self.n_x = input_vocab_size
        self.n_y = output_vocab_size
        self.initialize_params()
        
    def initialize_params(self):
        """
        Implements random initialization for the weights
        """
        self.Waa = np.random.randn(self.n_a, self.n_a) * np.sqrt(1/self.n_a)
        self.Wax = np.random.randn(self.n_a, self.n_x) * np.sqrt(1/self.n_a)
        self.Wya = np.random.randn(self.n_y, self.n_a) * np.sqrt(1/self.n_a)
        self.ba = np.zeros((self.n_a, 1))
        self.by = np.zeros((self.n_y, 1))
        
        # set up gradients of parameters to be filled in by the backward pass
        self.zero_out_gradients()
    
    def zero_out_gradients(self):
        """
        Zeroes out the gradients
        """
        self.dWaa = np.zeros((self.n_a, self.n_a))
        self.dWax = np.zeros((self.n_a, self.n_x))
        self.dWya = np.zeros((self.n_y, self.n_a))
        self.dba = np.zeros((self.n_a, 1))
        self.dby = np.zeros((self.n_y, 1))
        
    def initialize_adam_params(self):
        """
        Initializes the adam parameters
        """
        self.v_dWaa = np.zeros((self.n_a, self.n_a))
        self.v_dWax = np.zeros((self.n_a, self.n_x))
        self.v_dWya = np.zeros((self.n_y, self.n_a))
        self.v_dba = np.zeros((self.n_a, 1))
        self.v_dby = np.zeros((self.n_y, 1))

        self.s_dWaa = np.zeros((self.n_a, self.n_a))
        self.s_dWax = np.zeros((self.n_a, self.n_x))
        self.s_dWya = np.zeros((self.n_y, self.n_a))
        self.s_dba = np.zeros((self.n_a, 1))
        self.s_dby = np.zeros((self.n_y, 1))

        
    def softmax(self, Z):
        """
        Implements softmax

        Parameters:
            - Z: a numpy array of shape (n, m)
            
        Returns:
            - a numpy array of shape (n, m) with softmax applied to each
                column of Z
        """
        CLIP_THRESHOLD = 700
        clipped_Z = np.clip(Z, a_min=None, a_max=CLIP_THRESHOLD)
        temp = np.exp(clipped_Z)
        A = temp / np.sum(temp, axis=0)
        return A
        
    def forward_one_cell(self, a_prev, x_t):
        """
        Calculates the output of an RNN cell, given the previous hidden state
        and an input x_t
        
        Parameters:
            - a_prev: the previous hidden state, with shape (n_a, m),
                where m is the number of examples
            - x_t: the input to this cell as one-hot matrix, 
                with shape (n_x, m)
            
        Returns:
            - a_t: the current hidden state, with shape (n_a, m)
            - y_hat_t: the softmax output for the predicted output, with
                shape (n_y, m)
        """
        a_t = np.tanh(self.Waa @ a_prev + 
                      self.Wax @ x_t + 
                      self.ba)
        y_hat_t = self.softmax(self.Wya @ a_t + self.by)
        return a_t, y_hat_t
    
    def forward_pass(self, X):
        """
        Goes through one forward pass through all cells in the network
        
        Parameters:
            - X: the design matrix of shape (T_x, n_x, m), where T_x
                is the length of the inputs
                Is one-hot matrix
        """        
        T_x = X.shape[0]
        m = X.shape[-1]
        
        # initialize first hidden state, as well as list of hidden states
        # and predictions
        a_0 = np.zeros((self.n_a, m))
        self.a_s = [a_0]
        a_t = a_0
        self.y_hats = np.zeros((T_x, self.n_y, m))
        
        for t in range(T_x):
            x_t = X[t, :, :]
            a_t, y_hat_t = self.forward_one_cell(a_t, x_t)
            self.a_s.append(a_t)
            self.y_hats[t, :, :] = y_hat_t
        return
        
    def compute_cost(self, Y, Y_hat):
        """
        Computes the cross-entropy cost
        
        Parameters:
            - Y: the ground-truth labels of shape (T_y, n_y, m)
            - Y_hat: the predicted probabilities of each class of shape
                (T_y, n_y, m)
                
        Returns:
            - cost: the cross-entropy cost
        """
        EPSILON = 10e-8 # for numerical stability
        
        m = Y.shape[-1]
        ln_y_hat = np.log(Y_hat + EPSILON)
        cost = -np.einsum('ijk, ijk', ln_y_hat, Y)/m
        return cost
        
    def backward_one_cell(self, da_t_onward, a_t, x_t, y_hat_t, y_t,
        a_t_minus_one):
        """
        Calculates the gradients for one cell
        Assumes a tanh activation function
        
        Parameters:
            - da_t_onward: the part of dJ/da_t, the partial derivative of cost J
                wrt the current state, a_t, that affects the future time steps,
                has shape (n_a, m)
            - a_t: the hidden state for the current time step, has shape
                (n_a, m)
            - x_t: the input for the current time step, has shape (n_x, m)
            - y_hat_t: the softmax predictions for the current time step, 
                has shape (n_y, m)
            - y_t: the labels for the current time step, has shape (n_y, m)
            - a_t_minus_one: the hidden state for the previous time step, has
                shape (n_a, m)
                
        Returns:
            - da_t_minus_one: dJ/da_t-1, the partial derivative of cost J wrt the 
                previous hidden state, a_t-1
            - dx_t: dJ/dx_t, the partial derivative of cost J wrt the current
                input, x_t
            - dWax: dJ/dWax, the partial derivative of cost J wrt the parameter
                matrix Wax
            - dWaa: dJ/dWaa, the partial derivative of cost J wrt the parameter
                matrix Waa            
            - dWya: dJ/dWya, the partial derivative of cost J wrt the parameter
                matrix Wya
            - dba: dJ/dba, the partial derivative of cost J wrt the biases ba
            - dby: dJ/dby, the partial derivative of cost J wrt the biases by
        """
        m = a_t.shape[-1]
    
        # y_hat_t = softmax(z)
        dz = y_hat_t - y_t
        dWya = (dz @ a_t.T)/m
        da_t = self.Wya.T @ (dz) + da_t_onward
        
        # u = Wax(x_t) + Waa(a_t-1) + b_a, so a_t = tanh(u)
        du = da_t * (1 - np.square(a_t))
        dx_t = self.Wax.T @ du
        dWax = (du @ x_t.T)/m
        da_t_minus_one = self.Waa.T @ du
        dWaa = (du @ a_t_minus_one.T)/m
        dba = np.sum(du, axis=1, keepdims=True)/m
        dby = np.sum(dz, axis=1, keepdims=True)/m
        return da_t_minus_one, dx_t, dWax, dWaa, dWya, dba, dby
        
    def backward_pass(self, X, Y):
        """
        Goes through one backward pass through all cells in the network
        
        Parameters:
            - X: the design matrix of shape (T_x, n_x, m), where T_x
                is the length of the inputs
            - Y: the ground-truth labels of shape (T_x, n_y, m)                
        """        
        m = X.shape[-1]
        T_x = X.shape[0]
        
        # initialize the gradients
        self.zero_out_gradients()
        
        # initialize the variables for the very first cell in the backward
        # pass
        da_t_onward = np.zeros((self.n_a, m))
        #a_t = self.a_s[-1]
        #y_t = X[-1, :, :] # is <EOS>
        #x_t = X[-2, :, :]
        #y_hat_t = self.y_hats[-1, :, :]
        #a_t_minus_one = self.a_s[-2]
        
        for t in reversed(range(T_x)):
            a_t = self.a_s[t+1]
            y_t = Y[t, :, :]
            x_t = X[t, :, :]
            y_hat_t = self.y_hats[t, :, :]
            a_t_minus_one = self.a_s[t]
            da_t_onward, dx_t, dWax, dWaa, dWya, dba, dby = \
                self.backward_one_cell(da_t_onward, a_t, x_t, y_hat_t, y_t,
                                  a_t_minus_one)
            self.dWax += dWax
            self.dWaa += dWaa
            self.dWya += dWya
            self.dba += dba
            self.dby += dby
        return

    def params_to_vec(self, weights, biases):
        """
        Turns the weights and biases of the model into one vector
        for gradient checking
              
        Parametes:
            - weights: list of weights or the gradient of the weights in the
                order of Wax, Waa, and Wya
            - biases: list of biases or the gradient of the biases in the 
                order of ba and by
            
        Returns:
            - vec: a numpy array of shape (1, params), where params is the
                total number of parameters in the model
                The array has the parameters in the order
                Wax, Waa, ba, Wya, and by
        """
        vec = np.hstack((weights[0].reshape((1, -1)),
                         weights[1].reshape((1, -1)),
                         biases[0].reshape((1, -1)),
                         weights[2].reshape((1, -1)),
                         biases[1].reshape((1, -1))))
        return vec
        
    def vec_to_params(self, vec, n_x, n_a, n_y):
        """
        Turns a vector of parameters into weight and bias numpy arrays
        Undos the unrolling params_to_vec does
        
        Parameters:
            - vec: a numpy array of shape (1, params), where params is the
                total number of parameters in the model
                The array has the parameters in the order
                Wax, Waa, ba, Wya, and by
            - n_x: the input vocab size
            - n_a: the hidden state size
            - n_y: the output vocab size
                
        Returns:
            - Wax: one of the weight matrices used for calculating a_t,
                has shape (n_a, n_x)
            - Waa: the other weight matrix used for calculating a_t,
                has shape (n_a, n_a)
            - ba: the bias vector used for calculating a_t,
                has shape (n_a, 1)
            - Wya: the weight matrix used for calculating y_hat_t,
                has shape (n_y, n_a)
            - by: the bias vector used for calculating y_hat_t,
                has shape (n_y, 1)
        """   
        end_ind = n_a*n_x
        Wax = vec[:, :end_ind].reshape((n_a, n_x))
        start_ind = end_ind
        end_ind = start_ind+n_a*n_a
        Waa = vec[:, start_ind:end_ind].reshape((n_a, n_a))
        start_ind = end_ind
        end_ind = start_ind+n_a
        ba = vec[:, start_ind:end_ind].reshape((n_a, 1))
        start_ind = end_ind
        end_ind = start_ind+n_y*n_a
        Wya = vec[:, start_ind:end_ind].reshape((n_y, n_a))
        start_ind = end_ind
        end_ind = start_ind + n_y
        by = vec[:, start_ind:end_ind].reshape((n_y, 1))
        return Wax, Waa, ba, Wya, by
        
    def check_grads(self, X, Y, threshold, epsilon=10e-7):
        """
        Uses the 2-sided difference to compute the gradient
        This function is for checking the correctness of backprop.
        
        Parameters:
            - X: input to use
            - Y: the ground-truth labels           
            - threshold: how big a difference between dTheta and
                dTheta_approx is tolerated
            - epsilon: amount to add to each parameter        
        """
        # compute dTheta
        self.forward_pass(X)
        self.backward_pass(X, Y)
        dTheta = self.params_to_vec([self.dWax, self.dWaa, self.dWya], 
                                    [self.dba, self.dby])
        
        # compute dTheta_approx
        dTheta_approx = np.zeros(dTheta.shape)
        unrolled_params = self.params_to_vec([self.Wax, self.Waa, self.Wya], 
                                             [self.ba, self.by])        
        for i in range(unrolled_params.shape[1]):
            param_vec_plus_epsilon = unrolled_params.copy()
            param_vec_plus_epsilon[0, i] += epsilon
            
            param_vec_minus_epsilon = unrolled_params.copy()
            param_vec_minus_epsilon[0, i] -= epsilon
            
            Wax_plus, Waa_plus, ba_plus, Wya_plus, by_plus = \
                self.vec_to_params(param_vec_plus_epsilon, 
                                   self.n_x, self.n_a, self.n_y)
            
            Wax_minus, Waa_minus, ba_minus, Wya_minus, by_minus = \
                self.vec_to_params(param_vec_minus_epsilon,
                                   self.n_x, self.n_a, self.n_y)
            rnn_plus = RNN(self.n_a, self.n_x, self.n_y)
            rnn_plus.Wax, rnn_plus.Waa, rnn_plus.ba, rnn_plus.Wya, \
                rnn_plus.by = Wax_plus, Waa_plus, ba_plus, Wya_plus, by_plus
            rnn_plus.forward_pass(X)
            cost_plus = rnn_plus.compute_cost(X, rnn_plus.y_hats)
            
            rnn_minus = RNN(self.n_a, self.n_x, self.n_y)
            rnn_minus.Wax, rnn_minus.Waa, rnn_minus.ba, rnn_minus.Wya, \
                rnn_minus.by = Wax_minus, Waa_minus, ba_minus, Wya_minus, \
                by_minus  
            rnn_minus.forward_pass(X)
            cost_minus = rnn_minus.compute_cost(X, rnn_minus.y_hats)

            dTheta_approx[0, i] = \
                (cost_plus - cost_minus)/(2*epsilon)  
        
        difference = np.linalg.norm(dTheta_approx - dTheta) / (
            np.linalg.norm(dTheta_approx) + np.linalg.norm(dTheta))   
        if difference < threshold:
            print('Backprop and 2-sided difference are about equal')
        else:
            print('Backprop and 2-sided difference NOT equal',
                  'difference of:', difference)
            print(dTheta_approx - dTheta)
            Wax_diffs, Waa_diffs, ba_diffs, Wya_diffs, by_diffs = \
                self.vec_to_params(dTheta_approx-dTheta, 
                                   self.n_x, self.n_a, self.n_y)
            print('Wax diffs:', Wax_diffs)
            print('Waa diffs:', Waa_diffs)
            print('ba diffs:', ba_diffs)
            print('Wya diffs:', Wya_diffs)
            print('by diffs:', by_diffs)
            
            
    def update_parameters(self, learning_rate=0.01, max_grad=50):
        """
        Implements gradient descent to update the weights and biases
        
        Parameters:
            - learning_rate: the learning rate, alpha
            - max_grad: the maximum value to clip gradients to
        """        
        # clip gradients to avoid exploding gradients                
        self.Wax -= learning_rate*np.clip(self.dWax, a_min=-max_grad, 
                                          a_max=max_grad)
        self.Waa -= learning_rate*np.clip(self.dWaa, a_min=-max_grad, 
                                          a_max=max_grad)
        self.ba -= learning_rate*np.clip(self.dba, a_min=-max_grad, 
                                         a_max=max_grad)
        self.Wya -= learning_rate*np.clip(self.dWya, a_min=-max_grad, 
                                          a_max=max_grad)
        self.by -= learning_rate*np.clip(self.dby, a_min=-max_grad, 
                                         a_max=max_grad) 
        
    def update_params_adam(self, beta_1, beta_2, t, learning_rate=0.01,
        max_grad=50):
        """
        Updates the parameters using Adam optimization
        
        Parameters:
            - beta_1: the "momentum" hyperparameter
            - beta_2: the "RMSprop" hyperparameter
            - t: the time step, must be greater than 0
            - learning_rate: the learning rate, alpha            
            - max_grad: the maximum value to clip gradients to
        """
        EPSILON = 10e-08
        
        if not max_grad is None:            
            dWax = np.clip(self.dWax, a_min=-max_grad, a_max=max_grad)
            dWaa = np.clip(self.dWaa, a_min=-max_grad, a_max=max_grad)
            dWya = np.clip(self.dWya, a_min=-max_grad, a_max=max_grad)
            dba = np.clip(self.dba, a_min=-max_grad, a_max=max_grad)
            dby = np.clip(self.dby, a_min=-max_grad, a_max=max_grad)
        else:
            dWax = self.dWax
            dWaa = self.dWaa
            dWya = self.dWya
            dba = self.dba
            dby = self.dby
        self.v_dWax = beta_1*self.v_dWax + (1-beta_1)*dWax
        self.v_dWaa = beta_1*self.v_dWaa + (1-beta_1)*dWaa
        self.v_dWya = beta_1*self.v_dWya + (1-beta_1)*dWya
        self.v_dba = beta_1*self.v_dba + (1-beta_1)*dba
        self.v_dby = beta_1*self.v_dby + (1-beta_1)*dby
    
        self.s_dWax = beta_2*self.s_dWax + (1-beta_2)*np.square(dWax)
        self.s_dWaa = beta_2*self.s_dWaa + (1-beta_2)*np.square(dWaa)
        self.s_dWya = beta_2*self.s_dWya + (1-beta_2)*np.square(dWya)
        self.s_dba = beta_2*self.s_dba + (1-beta_2)*np.square(dba)
        self.s_dby = beta_2*self.s_dby + (1-beta_2)*np.square(dby)
        
        v_corrected_dWax = self.v_dWax/(1-beta_1**t)
        v_corrected_dWaa = self.v_dWaa/(1-beta_1**t)
        v_corrected_dWya = self.v_dWya/(1-beta_1**t)
        v_corrected_dba = self.v_dba/(1-beta_1**t)
        v_corrected_dby = self.v_dby/(1-beta_1**t)

        s_corrected_dWax = self.s_dWax/(1-beta_2**t)
        s_corrected_dWaa = self.s_dWaa/(1-beta_2**t)
        s_corrected_dWya = self.s_dWya/(1-beta_2**t)
        s_corrected_dba = self.s_dba/(1-beta_2**t)
        s_corrected_dby = self.s_dby/(1-beta_2**t)
        
        self.Wax -= \
            learning_rate*v_corrected_dWax/(np.sqrt(s_corrected_dWax)+EPSILON)
        self.Waa -= \
            learning_rate*v_corrected_dWaa/(np.sqrt(s_corrected_dWaa)+EPSILON)
        self.Wya -= \
            learning_rate*v_corrected_dWya/(np.sqrt(s_corrected_dWya)+EPSILON)
        self.ba -= \
            learning_rate*v_corrected_dba/(np.sqrt(s_corrected_dba)+EPSILON)            
        self.by -= \
            learning_rate*v_corrected_dby/(np.sqrt(s_corrected_dby)+EPSILON)            
        return
            
    def train(self, X, Y, t, beta_1=.9, beta_2=.999, 
        iterations=10, learning_rate=0.01, verbose=0,
        num_iter_msg=50, max_grad=50):
        """
        Runs several iterations, each consisting of:
            - a forward pass
            - a backward pass
            - updating the parameters onece
            
        Parameters:
            - X: the one-hot design matrix of shape (T_x, n_x, m), where T_x
                is the length of the inputs
            - Y: the one-hot targets in a matrix of shape (T_x, n_y, m)
            - t: the time step to pass to update_params_adam; must be
                greater than 0
            - beta_1: the beta_1 hyperparamter to for adam
            - beta_2: the beta_2 hyperparamter to for adam
            - iterations: the number of iterations to run
            - learning_rate: the learning rate to use for gradient descent
            - verbose: set to 0 for no messages, 1 to see the costs
                every num_iter_msg iterations
            - num_iter_msg: if verbose is set to 1, show the costs every
                num_iter_msg iterations
            - max_grad: the maximum value to clip gradients to                
        """
        for i in range(iterations):
            self.forward_pass(X)
            self.backward_pass(X, Y)
            self.update_params_adam(beta_1, beta_2, t, 
                learning_rate=learning_rate, max_grad=max_grad)
            if i % num_iter_msg == 0 and verbose == 1:
                cost = self.compute_cost(Y, self.y_hats)
                print('Iteration %s, Cost: %.4f' % (i, cost))
        return               
        
    def train_mini_batch(self, X, Y, epochs, mini_batch_size,
                         learning_rate=0.01, beta_1=.9, beta_2=.999,
                         num_iter_msg=50, max_grad=50, shift_X=False):
        """
        Runs mini-batch gradient descent
        
        Parameters:
            - X: the inputs as a list of lists, where each embedded list
                represents one example
            - Y: the targets as a list of lists, where each embedded list
                represents one example
            - epochs: the number of times to go through the training set X
            - mini_batch_size: the size of each mini-batch
            - learning_rate: the learning rate to use for gradient descent
            - beta_1: the beta_1 hyperparamter to for adam
            - beta_2: the beta_2 hyperparamter to for adam
            - num_iter_msg: if an integer, prints the overall cost
                after every show_cost_every_num_iter of iterations
            - max_grad: the maximum value to clip gradients to    
            - shift_X: if True, shifts the sequences in X to the left by
                1 time step, padding with zeros at the start
        """  
        iter = 1
        self.initialize_adam_params()
        
        for i in range(epochs):
            # get mini-batches
            mini_batches_X, mini_batches_Y = \
                get_mini_batches(X, Y, mini_batch_size)                             
        
            # iterate forward and backward with mini-batches
            for j in range(len(mini_batches_X)):
                # get one-hot encodings of X and Y
                X_oh = get_one_hot(mini_batches_X[j], self.n_x, self.n_x-1,
                    shift_left=shift_X)
                Y_oh = get_one_hot(mini_batches_Y[j], self.n_y, self.n_y-1)   
                
                self.train(X_oh, Y_oh, iter, beta_1, beta_2, 1, learning_rate, 
                           0, max_grad = max_grad)
                if not num_iter_msg is None and \
                    iter % num_iter_msg == 0:
                    X_oh_complete = get_one_hot(X, self.n_x, self.n_x-1,
                        shift_left=shift_X)
                    Y_oh_complete = get_one_hot(Y, self.n_y, self.n_y-1)
                    self.forward_pass(X_oh_complete)
                    cost = self.compute_cost(Y_oh_complete, self.y_hats)
                    print('Iteration: %s, Cost: %.4f' % (iter, cost))
                iter += 1
                
    def generate_sequence(self, term_output, max_length=100):
        """
        Randomly generates a sequence
        
        Parameters:
            - term_output: the index of the last character to generate; 
                once generated, the whole sequence is returned
            - max_length: the maximum length sequence to generate
                
        Returns:
            - seq: a list of randomly generated elements
        """
        
        # initialize first hidden state, as well as list of hidden states
        # and predictions
        a_0 = np.zeros((self.n_a, 1))        
        a_t = a_0
        x_t = np.zeros((self.n_x, 1))
        y_hat_t = None
        seq = []
        possible_inds = np.arange(self.n_x)
        
        while (y_hat_t != term_output and len(seq) < max_length):            
            a_t, y_hat_t = self.forward_one_cell(a_t, x_t)
            
            # convert from one-hot encoding before adding to seq
            y_hat_t = np.random.choice(possible_inds, p = y_hat_t.ravel())
            x_t = np.zeros((self.n_x, 1))
            x_t[y_hat_t,0] = 1
            seq.append(y_hat_t)            
        return seq

def get_one_hot(seqs, vocab_size, pad_ind, shift_left=False):
    """
    Takes a list of lists and converts them to one-hot arrays
    
    Parameters:
        - seqs: a list of lists, where the embedded lists will be converted
            to one-hot arrays
        - vocab_size: how large the vocabulary size is
        - pad_ind: the index to pad shorter sequences with
        - shift_left: if True, pads the sequence will zeros at the start
            and shifts over to the left by 1
        
    Returns:
        - seqs_oh: a one-hot encoding of seqs of size (T_x, n_x, m),
            where T_x is the length of the longest sequence in seqs,
            n_x is the vocabulary size, and m is the number of sequences
            in seqs
    """
    lens = [len(l) for l in seqs]
    max_T_x = max(lens)
    padded_inds = [np.pad(l, (0, max_T_x - len(l)), 'constant',
                          constant_values=(pad_ind)) for l in seqs]
    padded_inds = np.hstack(padded_inds)
    m = len(seqs)
    temp = np.zeros((max_T_x, vocab_size, m))
    T_x_inds = list(range(max_T_x)) * m    
    ex_inds = [x // max_T_x for x in range(m*max_T_x)]
    temp[T_x_inds, padded_inds, ex_inds] = 1
    seqs_oh = temp
    if shift_left:
        left_padded_seqs_oh = np.concatenate(
            (np.zeros((1, vocab_size, m)), seqs_oh), axis=0)
        seqs_oh = left_padded_seqs_oh[:max_T_x, :, :]
    return seqs_oh

def get_mini_batches(X, Y, mini_batch_size):
    """
    Returns a list of mini-batches of X and Y
    
    Parameters:
        - X: the inputs as a list of lists, where each embedded list
            represents one example
        - Y: the targets as a list of lists, where each embedded list
            represents one example
            The length of Y should match length of X
        - mini_batch_size: the size of each mini-batch
        
    Returns:
        - mini_batches_X: a list of mini-batches of X 
        - mini_batches_Y: a list of mini-batches of Y        
    """
    m = len(X)
    mini_batches_X = []
    mini_batches_Y = []
    nda_X = np.array(X)
    nda_Y = np.array(Y)
    rand_indices = np.random.permutation(m)
    shuffled_X = nda_X[rand_indices]
    shuffled_Y = nda_Y[rand_indices]
    count = 0
    for i in range(m // mini_batch_size):
        mini_batches_X.append(
            shuffled_X[count:(count+mini_batch_size)].tolist())
        mini_batches_Y.append(
            shuffled_Y[count:(count+mini_batch_size)].tolist())
        count += mini_batch_size
    if m % mini_batch_size != 0:
        mini_batches_X.append(shuffled_X[count:].tolist())
        mini_batches_Y.append(shuffled_Y[count:].tolist())
    return mini_batches_X, mini_batches_Y 
    
def un_one_hot(seqs_oh):
    """
    Converts a one-hot tensor to a list of numbers
    Undoes get_one_hot, except the padding is not removed
    
    Parameters:
        - seqs_oh: the one-hot tensor to convert to a list of numbers
            Is of size (T_x, n_x, m), where T_x is the length of the sequences
            in seqs_oh, n_x is the vocabulary size, and m is 
            the number of sequences in seqs_oh
        
    Returns:
        - seqs: the corresponding list of lists of seqs_oh
    
    """
    seqs = np.argmax(seqs_oh , axis=1)
    seqs = np.split(seqs, seqs.shape[1], axis=1)
    seqs = [a.flatten().tolist() for a in seqs]
    return seqs
    
def un_one_hot_2d(seqs_oh):
    """
    Converts a one-hot matrix to a list of numbers
    
    Parameters:
        - seqs_oh: the one-hot matrix to convert to a list of numbers
            Is of size (n_x, 1), where n_x is the vocabulary size, and
        
    Returns:
        - seqs: the corresponding list numbers in seqs_oh   
    """
    seqs = np.argmax(seqs_oh, axis=0)[0]
    return seqs
    