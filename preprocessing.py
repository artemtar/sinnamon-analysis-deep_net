import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import random
import pickle
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    with open(pos, 'r') as f:
        data = f.readlines()
        for l in data:
            words = word_tokenize(l)
            lexicon.append(words)

    with open(neg, 'r') as f:
        data = f.readlines()
        for l in data:
            words = word_tokenize(l)
            lexicon.append(words)

    word_bucket = []
    for l in lexicon:
        for w in l:
            word_bucket.append(lemmatizer.lemmatize(w.lower()))
    word_bucket = nltk.FreqDist(word_bucket)
    # removing noise

    noise = stopwords.words('english')
    for w in noise:
        if w in word_bucket:
            del word_bucket[w]
    additional_noise = [',', 'I', 'the', '``', '.', '\'\'', 'The', '(', ')']
    for w in additional_noise:
        if w in word_bucket:
            del word_bucket[w]
    return_bucket = []
    # common words only

    for v in word_bucket:
        if word_bucket[v] > 50:
            return_bucket.append(v)
    return return_bucket

# attaching lables and creating featureset
def handle_samples(sample, bag, classification):

    featureset = []
    with open(sample, 'r') as f:
        data = f.readlines()
        for l in data:
            features = np.zeros(len(bag))
            words = word_tokenize(l)
            for w in words:
                if w.lower() in bag:
                    index = bag.index(w.lower())
                    features[index] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_data(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    pos_sample = handle_samples(pos, lexicon, [1, 0])
    neg_sample = handle_samples(neg, lexicon, [0, 1])
    features = pos_sample + neg_sample
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(len(features) * test_size)
    train = features[:-testing_size]
    test = features[-testing_size:]

    train_x = list(train[:,0])
    train_y = list(train[:,1])
    test_x = list(test[:,0])
    test_y = list(test[:,1])
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_data("./data/pos.data", "./data/neg.data")
    with open('./data/train_and_test.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
