import tensorflow as tf
import numpy as np
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPool1D, \
    UpSampling1D, Flatten, Reshape
import json
from Project.fake_classification import load_bow, pad_seq
from sklearn.metrics import confusion_matrix


def to_onehot(x, steps, num_class):
    out = np.zeros([len(x), steps, num_class])
    for idx in range(len(x)):
        for step in range(steps):
            out[idx, step, int(x[idx, step])] = 1

    return out


class EmbeddingAutoEncoder:
    def __init__(self, bow, step, z_sz=64):
        # feature_sz is number of words in BOW
        self.bow = bow
        self.z_sz = z_sz
        self.feature_sz = len(bow.word_index)
        self.step = step
        self.ae, self.encoder = self._build_encoder()
        print(self.ae.trainable_variables[0] == self.encoder.trainable_variables[0])
        self.ae.summary()
        self.ae.compile(optimizer='adam', loss='mse')

    def _build_encoder(self):
        x = Input(shape=(self.step, self.feature_sz+1))

        # Encoder
        h = Conv1D(256, 16, padding='same', activation='relu')(x)
        h = MaxPool1D(5, padding='same')(h)
        h = Conv1D(128, 8, padding='same', activation='relu')(h)
        h = MaxPool1D(2, padding='same')(h)
        h = Flatten()(h) # None, 128
        _sz = h.shape[1]
        h = Dense(self.z_sz)(h)

        # Decoder
        out = h
        out = Dense(_sz, activation='relu')(out)
        out = Reshape((-1, 128))(out)  # Be careful with the shape
        out = Conv1D(128, 4, padding='same', activation='relu')(out)
        out = UpSampling1D(2)(out)
        out = Conv1D(self.feature_sz + 1, 16, padding='same')(out)
        out = UpSampling1D(5)(out)

        ae = Model(x, out)
        encoder = Model(x, h)

        return ae, encoder

    def train(self, x):
        x = self.bow.texts_to_sequences(x)
        x = pad_seq(x, self.step)
        x = to_onehot(x, self.step, self.feature_sz+1)

        self.ae.fit(x, x, batch_size=32, epochs=1)




