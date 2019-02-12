# clean_cities_data.py
import pandas as pd

CITY_GEO_ID_LEN = 16
NAME_DELIM = 'ZZZZZ'
DATA_FILEPATH = \
    'us_cities_data/DEC_10_SF1_GCTPH1.US13PR/DEC_10_SF1_GCTPH1.US13PR.csv'
# need encoding='latin_1' for accented vowels, like in San Jose
census = pd.read_csv(DATA_FILEPATH, 
    header=0, index_col=False, encoding='latin_1')

# filter to just the cities
census['len_geo_id'] = census['GCT_STUB.target-geo-id'].apply(len)
cities = census[census['len_geo_id'] == CITY_GEO_ID_LEN]

cities['city_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'city' in x)
cities['town_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'town' in x)
cities['cdp_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'CDP' in x)
cities['muni_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'municipality' in x)
cities['borough_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'borough' in x)
cities['village_bool'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: 'village' in x)
cities['include'] = \
    cities['city_bool'] | cities['town_bool'] | cities['cdp_bool'] | cities['muni_bool'] | cities['borough_bool'] | cities['village_bool']
cities = cities[cities['include'] == True]

cities['GCT_STUB.display-label.1'].replace(regex={'city': NAME_DELIM, 
    'town': NAME_DELIM, 'CDP': NAME_DELIM, 'municipality': NAME_DELIM, 
    'borough': NAME_DELIM, 'village': NAME_DELIM}, inplace=True)
cities['names'] = \
    cities['GCT_STUB.display-label.1'].apply(lambda x: x.split(NAME_DELIM)[0])


city_names = cities['names']
city_names = city_names.apply(str.rstrip)
city_names = city_names.drop_duplicates()
city_names.to_csv('us_cities_data/city_names.txt', 
    header=False, index=False)