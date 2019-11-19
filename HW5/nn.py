import numpy as np


class NN:
    def __init__(self, hidden_unit):
        self.hidden_unit = hidden_unit
        self.cost = None
        self.wh = None
        self.wo = None
        self.bh = None
        self.bo = None

    def build(self, features):

        self.wh = np.random.rand(features, self.hidden_unit)
        self.wo = np.random.rand(self.hidden_unit, 1)
        self.bh = np.random.rand(self.hidden_unit)
        self.bo = np.random.rand(1)

    @staticmethod
    def sig(x):
        return 1/(1 + np.exp(-x))

    def d_sig(self, x):
        return self.sig(x) * (1 - self.sig(x))

    def forward(self, x):
        wh, wo, bh, bo = self.wh, self.wo, self.bh, self.bo

        zh = np.dot(x, wh) + bh
        ah = self.sig(zh)
        zo = np.dot(ah, wo) + bo
        ao = self.sig(zo)

        return [ao, zo, ah, zh]

    def backward(self, x, y):
        y_hat, zo, ah, zh = self.forward(x)
        self.cost = - y * np.log(y_hat) - (1 - y)*np.log(1 - y_hat)

        delta_o = ((y_hat-y)/y_hat/(1-y_hat) * self.d_sig(zo))
        delta_h = np.dot(delta_o, self.wo.T) * self.d_sig(zh)

        dbo = delta_o
        dbh = delta_h

        dwo = np.dot(ah.reshape(-1, 1), delta_o.reshape(1, -1))
        dwh = np.dot(x.reshape(-1, 1), delta_h.reshape(1, -1))   # 1 at a time

        return dbo, dbh, dwo, dwh

    def train(self, x, y, lr=0.1):
        dbo, dbh, dwo, dwh = self.backward(x, y)
        self.wh -= lr * dwh
        self.wo -= lr * dwo
        self.bh -= lr * dbh
        self.bo -= lr * dbo

        return self.cost

    def inference(self, x):
        return self.forward(x)[0]


