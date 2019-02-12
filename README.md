# Welcome to Ballott Falls: Using RNN's for Language Modeling

This project randomly generates US city names using a recurrent neural network.

## Data
To get the names of American cities and towns, I used data from the 2010 census on the US Census Bureau's website.  Specifically, I used the ["Population, Housing Units, Area, and Density: 2010 - United States -- Places by State; and for Puerto Rico" table](https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=DEC_10_SF1_GCTPH1.US13PR&prodType=table).  After downloading the table of almost 30,000 cities, towns, etc., I filtered the data to include just the cities, towns, census-designated places, municipalities, boroughs, and villages.  Also, I removed any duplicate names, leaving about 19,000 unique names.  The data-cleaning code is in the `clean_cities_data.py` file in the `src` folder.  The `city_names.txt` file contains the cleaned list of place names.

## The Recurrent Neural Network

I implemented a recurrent neural network using Python and numpy, and the code for the RNN is in `src/rnn_lang_modeling.py`.  This RNN is a single-layer, unidirectional RNN.  Included functionality includes training using mini-batches, optimizing the parameters using Adam, and generating new sequences.

The file also includes the helper function `get_one_hot` to convert a list of sequences into one-hot arrays.

## Training/Hyperparameter Tuning
To train the network, I first converted each city name from `city_names.txt` into a list of numbers.  Each character is mapped to a unique number.  For example, "Akron" would be represented as `[13, 23, 30, 27, 26]` since the letter "a" is mapped to the number 13, "k" is mapped to the number 23, and so on.  I put each city's list of numbers into a list.  Since there is some alphabetical ordering in the data, I shuffled the list of city names before passing it to the RNN.  All of this is done in `load_cities_data.py`.

To create an RNN object, I used the following code:
```
import rnn_lang_modeling as rnn
import load_cities_data as lcd
city_seqs, char_to_ind, ind_to_char, n_x = lcd.get_cities_data()
rnn_1 = rnn.RNN(162, n_x, n_x)
```

The third line stores the list of list of numbers in `city_seqs`, a dictionary mapping characters to integers in `char_to_ind`, a dictionary mapping integers to characters in `ind_to_char`, and the number of unique characters in `n_x`.  Then an `RNN` object is created with a hidden state of size 162, an input vocabulary size of `n_x`, and an output vocabulary size of `n_x`.

To train the RNN, I used the following code:
```
rnn_1.train_mini_batch(city_seqs, city_seqs, epochs=20, mini_batch_size=128,
    learning_rate=0.00104, num_iter_msg=1000, max_grad=3,
    shift_X=True)
```

The first two arguments are the inputs and targets respectively.  The rest of the arguments are:
- `epochs=20` trains the RNN for 20 epochs
- `mini_batch_size=128` uses a mini-batch size of 128 
- `learning_rate=.00104` sets the learning rate to .00104
- `num_iter_msg=1000` makes the RNN print out the cost every 1,000 iterations
- `max_grad=3` clips the gradients to (-3, 3) before updating the parameters using Adam
- `shift_X=True` pads the X sequences with zeros one time-step on the left and shifts the X sequences left by one time-step

I tuned the hyperparamters by using a random search, which is implemented in the `cities_search_hyperparams.py` file.  Here, I used a 60/40 split for the training and validation sets.  For the learning rate, mini-batch size, hidden state size, and maximum gradient, I randomly chose a value and then trained using those hyperparameter values.  Then I calculated the validation cost for the trained model.  In total, I repeated this process 10 times and choose the set of hyperparameter values that achieved the lowest validation cost.  Those hyperparameter values are shown in the list above.

## Results: Generating New City Names

After training the RNN for 20 epochs, I generated several random city names, some more realistic than others:

- Keona
- Enwardota
- Ballott Falls
- Canney
- Enspare
- Vistleport
- Wint Hill
- Gotsen
- Whiceendton
- Kirmssblan
- Jangton
- Nowot
- Greerty
- Insvalbur
- Wrandford

(The actual generated names were all in lowercase; I capitalized the names for display purposes.)

The generated names show that the RNN has learned some patterns, such as town names ending in "-ford" or "-ton".

Overall, I was a little surprised at how plausible some of the fake names seem, and I'm ready to begin my new life in Wrandford.