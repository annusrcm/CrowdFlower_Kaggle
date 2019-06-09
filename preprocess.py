from datetime import datetime
import pandas as pd

from config import ParamConfig
from logger import Logger
from utils import clean_text


config = ParamConfig()


def preprocess_and_save_data(input_path, output_path):
    start = datetime.now()
    df = pd.read_csv(input_path).fillna("")
    Logger.log("Data pre processing starts")
    clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
    df = df.apply(clean, axis=1)
    df.to_csv(output_path,encoding='utf-8', index=False)

    Logger.log("{} finished pre processing in {}".format(input_path,datetime.now()-start))

if __name__ == "__main__":
    # pre process training csv
    preprocess_and_save_data(config.train_path, config.train_preprocessed_path)
    # pre process test csv
    preprocess_and_save_data(config.test_path, config.test_preprocessed_path)