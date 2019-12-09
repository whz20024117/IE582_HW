from Project.fake_classification import *
import pandas as pd
from urllib.parse import urlparse
import numpy as np
import pickle
import matplotlib.pyplot as plt


BOW = load_bow('./data/bow.json')
#model = FakeNewsModel(BOW, [512], [512, 256])

# load data
data = pd.read_csv('./data/data.csv')
data['agent_classification'] = data['URLs'].apply(lambda x: urlparse(x)[1])

headlines = data['Headline'].to_numpy()
labels = data['Label'].to_numpy()

shuffle = np.random.choice(len(labels), len(labels), replace=False)
headlines = headlines[shuffle]
labels = labels[shuffle]

split = int(0.8*len(labels))


ae = EmbeddingAutoEncoder(BOW, 30)
ae.train(headlines)
