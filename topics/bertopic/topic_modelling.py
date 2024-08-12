from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sys
import json
import random
import pandas as pd
import os
import glob
from bertopic import BERTopic
import numpy as np

limit = 0.9
path_prefix = "topics/"

print(f"Topics using BERTopics. Topics that cover {limit} of the data printed out.",flush=True)
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def read_data(path):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Walk through the directory
    for subdir, _, _ in os.walk(path):
        # Find all .tsv files in the current directory
        for file in glob.glob(os.path.join(subdir, '*.tsv')):
            # Read the file into a DataFrame
            df = pd.read_csv(file, sep='\t')
            print(f'Read {file} succesfully.', flush=True)
            # Append the DataFrame to the list
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Now `combined_df` contains data from all .tsv files
    print("All data read.", flush=True)
    return combined_df

def find_limit_index(supports, limit):
    """
    find the index that first covers limit (given) fraction of the data.
    E.g. [1,2,3,2,1,1] => returns 3 with limit 0.8, 4 with limit 0.9
    """
    cm = np.cumsum(supports)
    num_data = cm[-1]
    ind = [i for i in range(len(cm)) if cm[i] >= limit*num_data][0]
    return ind

    
df_all = read_data("/scratch/project_2009199/register-vs-genre/results/")

from ast import literal_eval
df2 = pd.DataFrame()
df2['genre_prediction'] = df_all['genre_prediction'].apply(literal_eval)
df2['register_prediction'] = df_all['register_prediction'].apply(literal_eval)
df2["text"] = df_all["text"]
df2 = df2.explode("register_prediction").explode("genre_prediction")
groups = df2.groupby(['register_prediction', 'genre_prediction'])

for name, d in groups:
    if name[0] not in ["HI","ID","IN", "IP"]:
        print(f'\n---------------------------\n{list(name)}', flush=True)
        data = d["text"].tolist()
        print(f'Support: {len(data)}', flush=True)
        random.shuffle(data)
        data = data[0:20000]   # This is to reduce runtime with some really large classes
        
        topic_model = BERTopic()
        try:
            topics, probs = topic_model.fit_transform(data)
        except:
            print(f"Cannot calculate topics for {name}")
            continue
        df = topic_model.get_topic_info()
        #ind = find_limit_index(np.array(df["Count"]), limit)
        path = path_prefix+"-".join(["".join(i.split(" ")).replace(",","") for i in name])+".csv"
        print(f"Saving to {path}", flush=True)
        df = df.drop("Representative_Docs", axis="columns")
        df.to_csv(path)
