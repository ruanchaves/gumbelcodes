import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
import re
import spacy

def parse(x):
    return ' '.join([y.text for y in nlp(x, disable=['parser', 'tagger', 'ner']) if y.is_alpha])

nlp = en_core_web_sm.load()
data['sentence'] = data['sentence'].apply(parse)

class Model(object):
    def __init__(self,
                max_features=20000):
        self.max_features = max_features

class Dataset(Model):
    def __init__(self, 
                data=['data/train.tsv', 'data/dev.tsv'],
                sep='\t',
                nlp=None,
                sentence_header='sentence',
                sentiment_header='label',
                len_header='len',
                **kwargs):
        super(Dataset, self).__init__(**kwargs)
        if isinstance(data, str):
            self.data = pd.read_csv(data, sep=sep)
        elif isinstance(data, list):
            self.data = pd.read_csv(data[0], sep=sep)
            for item in data[1:]:
                tmp = pd.read_csv(item, sep=sep)
                self.data = self.data.append(tmp)
        self.nlp = nlp
        self.sentence_header = sentence_header
        self.sentiment_header = sentiment_header
        self.tokenizer = None
        self.word_index = None

    def spacy_tokenize(self):
        self.data[sentence_header] = self.data[sentence_header].apply(parse)
        self.data[len_header] = self.data[sentence_header].apply(lambda x: len(str(x).split(' ')))
        self.sequence_length = data[len_header].max() + 1
        self.max_features = max_features

    def keras_train_test_split(self,
                split=' ',
                oov_token='<unw>',
                filters=' ')
    self.tokenizer = Tokenizer(num_words=self.max_features, split=split, oov_token=oov_token, filters=filters)
    self.tokenizer.fit_on_texts(self.data[self.sentence_header].values)
    X = self.tokenizer.texts_to_sequences(self.data[self.sentence_header].values)
    X = pad_sequences(X, self.sequence_length)
    y = pd.get_dummies(self.data[self.sentiment_label]).values

    self.word_index = self.tokenizer.word_index
    return train_test_split(X, y, test_size=0.3)

    def parse(x):
        return ' '.join([y.text for y in self.nlp(x, disable=['parser', 'tagger', 'ner']) if y.is_alpha])

class WordEmbedding(Dataset):
    def __init__(self,
                path='data/glove.6B.300d.txt',
                encoding='utf-8',
                dtype='float32',
                **kwargs):
        super(WordEmbedding, self).__init__(**kwargs)
        self.path = path
        self.encoding = encoding
        self.dtype = dtype
        self.embeddings_index = {}
        self.embedding_matrix = None
        self.num_words = None

    def read_embedding(self):
        f = open(self.path, encoding=self.encoding)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        word_index = tokenizer.word_index

    def build_embedding_matrix(self):
        self.num_words = min(self.max_features, len(self.word_index)) + 1
        self.embedding_matrix = np.zeros((self.num_words, self.embedding_dim))
        for word, i in word_index.items():
            if i > max_features:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # we found the word - add that words vector to the matrix
                self.embedding_matrix[i] = embedding_vector
            else:
                # doesn't exist, assign a random vector
                self.embedding_matrix[i] = np.random.randn(self.embedding_dim)

class SentimentModel(WordEmbedding):

    def __init__(self, **kwargs):
        super(SentimentModel, self).__init__(**kwargs)
        self.model = None

    def compile(self, units=2, trainable=False):
        self.model = Sequential()
        self.model.add(Embedding(self.num_words,
                            self.embedding_dim,
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=self.sequence_length,
                            trainable=trainable))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
        self.model.add(Bidirectional(CuDNNLSTM(32)))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=units, activation='softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    def evaluate(self, X_test, y_test):
        y_hat = model.predict(X_test)
        accuracy_score(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
        conf = confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
        tp = conf[0][0]
        fn = conf[0][1]
        fp = conf[1][0]
        tn = conf[1][1]

        matthews_correlation_coefficient = \
        ((tp * tn) - (fp * fn)) / ( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)  ) ** 0.5

        return {
            "Matthews correlation coefficient" : matthews_correlation_coefficient,
            "Accuracy score": accuracy_score
        }
    
class SentimentPipeline(SentimentModel):
    def __init__(self, **kwargs):
        super(SentimentPipeline, self).__init__(**kwargs)

    def execute(self):
        self.spacy_tokenize()
        self.keras_train_test_split()
        self.read_embedding()
        self.build_embedding_matrix()
        self.compile()
        return self.evaluate()
