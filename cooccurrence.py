import pandas as pd
import numpy as np

data = pd.read_csv("/scratch/project_2009199/register-vs-genre/register-and-genre/new_results/large_large_03_04.tsv", delimiter="\t", on_bad_lines='skip')

def flatten(matrix):
    return [i for row in matrix for i in eval(row)]

labels_genre = np.unique(flatten(data.genre_prediction.to_list()))
labels_register = np.unique(flatten(data.register_prediction.to_list()))
co = np.zeros((len(labels_genre), len(labels_register)),dtype=int)
for i,d in data.iterrows():
    gs = d.genre_prediction
    rs = d.register_prediction
    for g in eval(gs):
        for r in eval(rs):
            i,j = np.where(labels_genre==g)[0], np.where(labels_register==r)[0]
            co[i,j]+=1

#print(co)
import seaborn as sns

#heatmap=sns.heatmap(np.log(co+1), yticklabels=labels_genre, xticklabels=labels_register, cmap=sns.cubehelix_palette(start=.2,rot=-.3, as_cmap=True, reverse=True))
heatmap=sns.heatmap(np.log(co+1), yticklabels=labels_genre, xticklabels=labels_register, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True))
heatmap.figure.tight_layout()
fig = heatmap.get_figure()
fig.savefig("/scratch/project_2009199/register-vs-genre/register-and-genre/plots/large_large_03_04.png") 
