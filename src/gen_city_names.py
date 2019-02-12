# gen_city_names.py
import rnn_lang_modeling as rnn
import load_cities_data as lcd

city_seqs, char_to_ind, ind_to_char, n_x = lcd.get_cities_data()

rnn_1 = rnn.RNN(162, n_x, n_x)

rnn_1.train_mini_batch(city_seqs, city_seqs, epochs=20, mini_batch_size=128,
    learning_rate=0.00104, num_iter_msg=1000, max_grad=3,
    shift_X=True)

for i in range(15):                         
    fake_city = rnn_1.generate_sequence(char_to_ind['\n'])
    name = [ind_to_char[i] for i in fake_city]
    name = ''.join(name)                  
    print(name)