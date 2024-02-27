# getting word similarity
# requires weights from trained neural 
from collections import Counter
import math
import pickle

# Retrieve data from the saved file
word2index = {}
with open("saved_data.pkl", "rb") as file:
    word2index = pickle.load(file)

# print('typeof word2index', type(word2index))
# print('word2index',word2index['beautiful'])

# def similar(target='beautiful'):
#     target_index = word2index[target]
#     scores = Counter()
#     for word, index in
# similar()
