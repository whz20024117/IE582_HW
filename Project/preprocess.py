import pandas as pd
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import json


def main():
    data = pd.read_csv('data/data.csv')
    data['agent_classification'] = data['URLs'].apply(lambda x: urlparse(x)[1])

    headlines = data['Headline'].tolist()
    labels = data['Label'].tolist()

    BOW = Tokenizer()
    BOW.fit_on_texts(headlines)

    # Save Bag of Word
    with open('./data/bow.json', 'w') as f:
        json.dump(BOW.to_json(), f)

    return


if __name__ == '__main__':
    main()
