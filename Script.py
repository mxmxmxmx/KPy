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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 20000
maxlen = 200
embed_size = 128

train = pd.read_csv('Input/train.csv')
test = pd.read_csv('Input/test.csv')
submission = pd.read_csv('Input/sample_submission.csv')

y = train[train.columns[2:]].values
list_sentences_train = train['comment_text']
list_sentences_test = test['comment_text']

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_sentences_train = tokenizer.texts_to_sequences(list_sentences_train)
list_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)

x_t = pad_sequences(list_sentences_train, maxlen=maxlen)
x_test = pad_sequences(list_sentences_test, maxlen=maxlen)

totalNumWords = [len(one_comment) for one_comment in list_sentences_train]

inp = Input(shape = (maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(60, return_sequences=True, name = 'lstm_layer'))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(6, activation='sigmoid')(x)
model = Model(inputs = inp, outputs = x)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 32
epochs = 2
model.fit(x_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)
