import numpy as np
import pandas as pd
from utils import clean_text
from config import config

###############
## Load Data ##
###############
print("Load data...")

train_path = "/home/annu/Downloads/data/crowdflower-search-relevance/small_train.csv"

# dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
dfTrain = pd.read_csv("/home/annu/Downloads/data/crowdflower-search-relevance/small_train.csv").fillna("")

num_train = dfTrain.shape[0]
print("Done.")


######################
## Pre-process Data ##
######################
print("Pre-process data...")

## insert sample index
dfTrain["index"] = np.arange(num_train)
# dfTest["index"] = np.arange(num_test)

## one-hot encode the median_relevance
for i in range(config.n_classes):
    dfTrain["median_relevance_%d" % (i+1)] = 0
    dfTrain["median_relevance_%d" % (i+1)][dfTrain["median_relevance"]==(i+1)] = 1
    
## query ids
qid_dict = dict()
for i,q in enumerate(np.unique(dfTrain["query"]), start=1):
    qid_dict[q] = i
    
## insert query id
# dfTrain["qid"] = map(lambda q: qid_dict[q], dfTrain["query"])

## clean text
clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
dfTrain = dfTrain.apply(clean, axis=1)
# dfTest = dfTest.apply(clean, axis=1)

print(dfTrain)

print("Done.")


###############
## Save Data ##
###############
print("Save data...")

train_preprocessd = "/home/annu/Downloads/data/crowdflower-search-relevance/small_train_pp1.csv"
dfTrain.to_csv(train_preprocessd,encoding='utf-8', index=False)


print("Done.")
