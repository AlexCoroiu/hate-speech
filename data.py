import json
from random import shuffle
import pickle
import os

d = 7178

def get(json_file, label):
    with open(json_file, 'r') as file:
        return [(json.loads(line)['text'], label) for line in file]

#get data
sexism = get('data/sexism.json', 'sexism')
racism = get('data/racism.json', 'racism')
neither = get('data/neither.json', 'neither')

#construct training and testing sets
training = sexism + neither[:d]
shuffle(training)
print(training[0])
testing = racism + neither[(d + 1):]
shuffle(testing)

#save training and testing data
tr_file = 'training.txt'
te_file = 'testing.txt'

if os.path.exists(tr_file):
    os.remove(tr_file)
with open('training.txt', 'w+b') as tr:
    pickle.dump(training, tr)

if os.path.exists(te_file):
    os.remove(te_file)
with open('testing.txt', 'w+b') as te:
    pickle.dump(testing, te)

