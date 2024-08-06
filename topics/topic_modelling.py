from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import sys
import json
import random
import pandas as pd
import os
import glob

#print(sys.argv)
#input = sys.argv[1]
#amount = int(sys.argv[2])
topics = int(sys.argv[1])
num_words = int(sys.argv[2])
amount=2000
limit=20

print(f'Topic analysis for genre-register labelled data. \nUsing {topics} topics from <= {amount} suffled documents, with classes where num_examples < {limit} is marked separately.')
print('Stopwords from nltk english and wordnetlemmatizer. Lowercased and punctuation cleaned.')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(lemma.lemmatize(ch) for ch in stop_free if ch not in exclude)
    return punc_free

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


df_all = read_data("/scratch/project_2009199/register-vs-genre/results/")

from ast import literal_eval
df2 = pd.DataFrame()
df2['genre_prediction'] = df_all['genre_prediction'].apply(literal_eval)
df2['register_prediction'] = df_all['register_prediction'].apply(literal_eval)
df2["text"] = df_all["text"]
df2 = df2.explode("register_prediction").explode("genre_prediction")
groups = df2.groupby(['register_prediction', 'genre_prediction'])

for name, d in groups:
    print(f'---------------------------\nNow in {name}')
    data = d["text"].tolist()
    if len(data) < limit:
        print(f'Not enough data for {name}, num_docs = {len(data)} < {limit} = limit.')
    else:
        print(f'Support is {len(data)}')
    random.shuffle(data)
    data = data[0:amount]

    doc_clean = [clean(doc).split() for doc in data]

    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    from pprint import pprint
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=topics, id2word =dictionary, passes=30)
    pprint(ldamodel.print_topics(num_topics=topics, num_words=num_words))
    print("")
print("perpelixity...?")
print(ldamodel.log_perplexity(data))