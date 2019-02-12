# load_cities_data.py
import numpy as np

def get_cities_data():
    """
    Loads and returns the city names data
    
    Returns:
        - city_seqs: a list of lists, representing the city names
        - char_to_ind: a dictionary mapping each character to its respective
            index
        - ind_to_char: a dictionary mapping each index to its respective
            character
        - vocab_size: the number of characters in the data
    """
    cities = open('us_cities_data/city_names.txt', 'r', encoding='utf-8')
    characters = [' ', '"', "'", '(', ')', ',', '-', '.', '1', '2', '5', 
                  '6', '7', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
                  'w', 'x', 'y', 'z', 'ñ', '\n']
                  
    char_to_ind = {char: ind for (ind, char) in enumerate(characters)}
    ind_to_char = {ind: char for (ind, char) in enumerate(characters)}

    city_seqs = []
    for line in cities:
        city_seqs.append([char_to_ind[c] for c in line.lower()])
    cities.close()

    vocab_size = len(characters)
    np.random.shuffle(city_seqs)
    return city_seqs, char_to_ind, ind_to_char, vocab_size