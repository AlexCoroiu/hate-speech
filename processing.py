import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tdif = False

#reading training and testing data
with open('training.txt','rb') as tr:
    training = pickle.load(tr)
with open('testing.txt','rb') as te:
    testing = pickle.load(te)

def process(text):
    tokenizer = nltk.TweetTokenizer(False, True, True)
    return tokenizer.tokenize(text)
#remove stop words (other useless words)
#stemming and lemmatization

#prepare labels
train_data = [process(entry[0]) for entry in training]
test_data = [process(entry[0]) for entry in testing]

#'neither'/'hate'
#0/1
train_labels = [0 if entry[1] == 'neither' else 1 for entry in training]
test_labels = [0 if entry[1] == 'neither' else 1 for entry in testing]

def dummy(set):
    return set

#builds vocab
if tdif:
    vectorizer = TfidfVectorizer(use_idf = True, lowercase = False, tokenizer = dummy, preprocessor = dummy)
else:
    vectorizer = CountVectorizer(lowercase= False, tokenizer=dummy, preprocessor= dummy)


vocab = vectorizer.fit(train_data).vocabulary_.keys()
print(vocab.__len__(),vocab)
test_vocab = vectorizer.fit(test_data).vocabulary_.keys()
print(test_vocab.__len__(),test_vocab)
oov = test_vocab - vocab
print(oov.__len__(),oov)

#13756
#11266
#6238

#vectorizer.fit(train_data)

#VECTORIZEZ
#train_vect = vectorizer.transform(train_data) #[] to be interpreted as one doc ==> [['a','b']] 1 elem list
#test_vect = vectorizer.transform(test_data)


