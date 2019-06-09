import pandas as pd
from utils import clean_text
from config import config

print("Load data...")

train_path = "/home/annu/Downloads/data/crowdflower-search-relevance/small_train.csv"

# dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
dfTrain = pd.read_csv("/home/annu/Downloads/data/crowdflower-search-relevance/small_train.csv").fillna("")

num_train = dfTrain.shape[0]
print("Pre-process data...")

clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
dfTrain = dfTrain.apply(clean, axis=1)

print("Save data...")

train_preprocessd = "/home/annu/Downloads/data/crowdflower-search-relevance/small_train_pp1.csv"
dfTrain.to_csv(train_preprocessd,encoding='utf-8', index=False)


print("Done.")
