import numpy
from nn import NN
import pandas as pd


model = NN(10)

data = pd.read_csv('xor.csv', header=None).to_numpy()
xs = data[:,:2]
ys = data[:,2:]
model.build(2)

for i in range(len(data)):
    print(model.backward(xs[i], ys[i]))