import json
import numpy as np
import pickle
from keras.models import load_model
from datetime import datetime

from logger import Logger
from utils import get_data_for_model
from config import ParamConfig


if __name__ == "__main__":

    Logger.log("PREDICTION STARTED")
    start = datetime.now()
    config = ParamConfig()
    X, ids = get_data_for_model(config.test_preprocessed_path, training_flag=False)

    with open(config.tokenizer_name, 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model(config.model_name)

    eval_data = tokenizer.texts_to_matrix(X, mode='tfidf')

    prediction = model.predict(eval_data)

    with open(config.encoder_name) as f:
        encoded_class = json.load(f)

    Logger.log("RELEVANCE -----> QUERY ID")
    for i in range(len(prediction)):
        class_pred = np.argmax(prediction[i])
        predicted_relevance = encoded_class[str(class_pred)]
        Logger.log("{} -----> {}".format(predicted_relevance, ids[i]))

    Logger.log("Prediction finished in : {}".format(datetime.now()-start))