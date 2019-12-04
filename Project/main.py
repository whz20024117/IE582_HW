from models import *
import pandas as pd
from urllib.parse import urlparse


BOW = load_bow()
model = FakeNewsModel(BOW, [512], [512, 256])

# load data
data = pd.read_csv('data/data.csv')
data['agent'] = data['URLs'].apply(lambda x: urlparse(x)[1])

headlines = data['Headline'].tolist()
labels = data['Label'].tolist()


model.train(headlines,labels)
