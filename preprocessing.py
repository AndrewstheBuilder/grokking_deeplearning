import sys
import pickle

f = open("reviews.txt")
raw_reviews = f.readlines()
f.close()

f = open("labels.txt")
raw_labels = f.readlines()
f.close()

tokens = list(map(lambda x: set(x.split(" ")), raw_reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if(len(word) > 0):
            vocab.add(word)
vocab = list(vocab) # why do this list() conversion?

word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
    # word2index will contain of the index of the word where it last occurred.
    # if(word in word2index):
    #     print('word is a duplicate:', word)
    #     print('index', i)

input_dataset = list() # what is the purpose of this data structure?
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)
#print('tokens',tokens[:1])

# Save data to a file using pickle
with open("saved_data.pkl", "wb") as file:
    pickle.dump(word2index, file)

# Intro to Embedding
# Build Neural Network here
import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

alpha, iterations = (0.01, 2) # What is alpha used for?
hidden_size = 100

weights_0_1 = 0.2*np.random.random((len(vocab)) ,hidden_size) - 0.1
weights_1_2 = 0.2*np.random.random(hidden_size , 1) - 0.1

correct, total = (0,0)
for iter in range(iterations):
    