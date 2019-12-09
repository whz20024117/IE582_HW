from Project.fake_classification import *
import pandas as pd
from urllib.parse import urlparse
import numpy as np
import pickle
import matplotlib.pyplot as plt


BOW = load_bow('./data/bow.json')
model = FakeNewsModel(BOW, [512], [512, 256])

# load data
data = pd.read_csv('./data/data.csv')
data['agent_classification'] = data['URLs'].apply(lambda x: urlparse(x)[1])

headlines = data['Headline'].to_numpy()
labels = data['Label'].to_numpy()

shuffle = np.random.choice(len(labels), len(labels), replace=False)
headlines = headlines[shuffle]
labels = labels[shuffle]

split = int(0.8*len(labels))

history = model.train(headlines[:split], labels[:split], epochs=15)
model.save()

np.save('./save/acc.npy', history.history['accuracy'])
np.save('./save/loss.npy', history.history['loss'])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


acc, cm = model.test(headlines[split:], labels[split:])
print(acc)
print(cm)
