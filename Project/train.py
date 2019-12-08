from models import *
import pandas as pd
from urllib.parse import urlparse
import numpy as np


BOW = load_bow()
model = FakeNewsModel(BOW, [512], [512, 256])

# load data
data = pd.read_csv('data/data.csv')
data['agent'] = data['URLs'].apply(lambda x: urlparse(x)[1])

headlines = data['Headline'].to_numpy()
labels = data['Label'].to_numpy()

shuffle = np.random.choice(len(labels), len(labels), replace=False)
headlines = headlines[shuffle]
labels = labels[shuffle]

split = int(0.8*len(labels))

model.train(headlines[:split], labels[:split], epochs=10)
acc, cm = model.test(headlines[split:], labels[split:])
print(acc)
print(cm)
