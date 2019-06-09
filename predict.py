import pandas as pd
import numpy as np
import pickle
from keras.models import load_model



test_processed = "/home/annu/Downloads/data/crowdflower-search-relevance/small_test.csv"


data = pd.read_csv(test_processed)

data["expanded_query"] = data["product_title"] + data["product_description"]
data = data.fillna("")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('my_model.h5')


test_data = tokenizer.texts_to_matrix(data["expanded_query"], mode='tfidf')

prediction = model.predict(test_data)

for pred in prediction:
    print(np.argmax(pred))