import tensorflow as tf
import numpy as np
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding
import json
from sklearn.metrics import confusion_matrix


# Some helper functions
def load_bow(path='./data/bow.json'):
    with open('./data/bow.json') as f:
        _bow = json.load(f)
    bow = tokenizer_from_json(_bow)

    return bow


def pad_seq(seqs, max_len):
    tmp = np.zeros(shape=(len(seqs), max_len), dtype=np.float)

    for idx, seq in enumerate(seqs):
        if len(seq) > max_len:
            seq = seq[:max_len]
        tmp[idx, :min(len(seq), max_len)] = seq

    return tmp


class FakeNewsModel:
    def __init__(self, bow, lstm_units, fc_units, emb_sz=256, max_len=30, lr=0.001):
        self.lstm_units = lstm_units
        self.bow = bow
        self.fc_units = fc_units
        self.emb_sz = emb_sz
        self.vocab_sz = len(bow.word_index)
        self.max_len = max_len
        self.model = self._build()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                           loss=tf.keras.losses.BinaryCrossentropy())
        self.model.summary()

    def _build(self):
        x = Input(shape=(self.max_len,))
        out = Embedding(self.vocab_sz + 1, self.emb_sz, mask_zero=True)(x)
        # LSTM layers

        if len(self.lstm_units)>1:
            for p in self.lstm_units[:-1]:
                out = LSTM(p, return_sequences=True, activation='relu')(out)
                out = Dropout(0.3)(out)
            out = LSTM(self.lstm_units[-1], activation='relu')(out)
        elif len(self.lstm_units)==1:
            out = LSTM(self.lstm_units[-1], activation='relu')(out)

        # out = LSTM(self.lstm_units[-1], activation='relu')(out)

        # FC layers
        for p in self.fc_units:
            out = Dense(p, activation='relu')(out)

        out = Dense(1, activation='sigmoid')(out)

        return Model(x, out)

    def train(self, data, label, batch_sz=32, epochs=20):
        x = self.bow.texts_to_sequences(data)
        x = pad_seq(x, self.max_len)
        y = np.array(label, dtype=np.float).reshape([-1, 1])

        return self.model.fit(x, y, batch_size=batch_sz, epochs=epochs)

    def predict(self, data):
        x = self.bow.texts_to_sequences(data)
        x = pad_seq(x, self.max_len)

        return self.model.predict(x)

    def test(self, data, label):
        x = self.bow.texts_to_sequences(data)
        x = pad_seq(x, self.max_len)
        y = np.array(label, dtype=np.float).flatten()

        pred = self.model.predict(x).flatten()
        pred = np.array(list(map(lambda p: 0 if p < 0.5 else 1, pred)))

        assert len(pred.shape) == 1
        assert pred.shape == y.shape

        acc = np.float(sum(pred == y))/float(len(pred))
        cm = confusion_matrix(y, pred)

        return acc, cm
