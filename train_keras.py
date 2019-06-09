# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import codecs


# For reproducibility
np.random.seed(1237)

# Source file directory
path_train = "/home/annu/Downloads/data/crowdflower-search-relevance/small_train_pp1.csv"


# We have training data available as dictionary filename, category, data
data = pd.read_csv(path_train)

data["expanded_query"] = data["query"] + " " + data["product_title"] + data["product_description"]
data = data.fillna("")
print(data.shape[0])

# print(data["expanded_query"])

num_labels = 4
vocab_size = 100
batch_size = 10
num_epochs = 2

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .8)

train_posts = data["expanded_query"][:train_size]
train_tags = data["median_relevance"][:train_size]
train_ids = data["id"][:train_size]


test_posts = data["expanded_query"][train_size:]
test_tags = data["median_relevance"][train_size:]
test_ids = data["id"][train_size:]

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

print(x_train.shape)
print(y_train.shape)

# y_train = train_tags
# y_test = test_tags

model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)

# Turn this on if you need to save model and tokenizer

# # creates a HDF5 file 'my_model.h5'
# model.model.save('my_model.h5')
#
# # Save Tokenizer i.e. Vocabulary
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])

text_labels = encoder.classes_

for i in range(5):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print(test_ids.iloc[i])
    print('Actual label: {}'.format(test_tags.iloc[i]))
    print("Predicted label: {} ".format(predicted_label))


y_pred = model.predict(x_test)
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(cnf_matrix)