import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk import TreebankWordTokenizer, wordnet
from nltk.corpus import stopwords
from datetime import datetime

from logger import Logger
from replacer import CsvWordReplacer
from config import ParamConfig

config = ParamConfig()


def text_preprocessor(x):
    toker = TreebankWordTokenizer()
    lemmer = wordnet.WordNetLemmatizer()
    x_cleaned = x.replace('/', ' ').replace('-', ' ').replace('"', '')
    x_cleaned = x_cleaned.lower()
    x_cleaned = re.sub("\d+", "", x_cleaned)
    tokens = toker.tokenize(x_cleaned)
    tokens = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join([lemmer.lemmatize(z) for z in tokens])


def clean_text(line, drop_html_flag=False):
    replacer = CsvWordReplacer(config.synonyms_csv)
    names = ["query", "product_title", "product_description"]
    for name in names:
        l = line[name]
        # clean html
        l = drop_html(l)
        # preprocess text
        l = text_preprocessor(l)
        # remove links
        l = re.sub(r'http:\\*/\\*/.*?\s', ' ', l)

        ## replace other words
        for k, v in config.replace_dict.items():
            l = re.sub(k, v, l)
        l = l.split(" ")

        ## replace synonyms
        l = replacer.replace(l)
        l = " ".join(l)
        line[name] = l
    return line


def drop_html(html):
    soup = BeautifulSoup(html, features="html5lib")
    for s in soup(['script', 'style']):
        s.decompose()
    return ' '.join(soup.stripped_strings)




def get_data_for_model(file_path, training_flag=True):
    data = pd.read_csv(file_path).fillna("")
    X = data["product_title"] + data["product_description"]
    Y_encoded = []
    if training_flag:
        Y = data["median_relevance"]
        for i in range(len(Y)):
            Y_encoded.append(int(Y[i]))
        return X,Y_encoded
    else:
        return X, data["id"]
