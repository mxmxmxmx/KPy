# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:09:40 2018

@author: Meng Xu
"""

# FOR THE CONTEST OF "TOXIC COMMENT"
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, concatenate, Flatten
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

max_features = 30000
maxlen = 200
embed_size = 300

train = pd.read_csv('../Toxic Comment Classification Challenge/Input/train.csv')
test = pd.read_csv('../Toxic Comment Classification Challenge/Input/test.csv')
submission = pd.read_csv('../Toxic Comment Classification Challenge/Input/sample_submission.csv')

y = train[train.columns[2:]].values
list_sentences_train = train['comment_text']
list_sentences_test = test['comment_text']

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_sentences_train = tokenizer.texts_to_sequences(list_sentences_train)
list_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)

x_t = pad_sequences(list_sentences_train, maxlen=maxlen)
x_test = pad_sequences(list_sentences_test, maxlen=maxlen)


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))
            
def get_model1():
    inp = Input(shape = (maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    
    x = Bidirectional(LSTM(60, return_sequences=True, name = 'lstm_layer'))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dropout(0.2)(conc)
    
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


def get_model2():
    inp = Input(shape = (maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model

model = get_model1()

#model = get_model2()
batch_size = 32
epochs = 5
#model.fit(x_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.fit(x_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)

model = get_model1()
X_tra, X_val, y_tra, y_val = train_test_split(x_t, y, train_size=0.90, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist=model.fit(X_tra, y_tra, batch_size=200, epochs=1, validation_data=(X_val, y_val),
                     callbacks=[RocAuc], verbose=1)


import numpy as np
wordsList = np.load('wordsList.npy')
