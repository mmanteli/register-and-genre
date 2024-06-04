import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datasets import load_dataset, Dataset
import numpy as np
import datasets
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
#from train_multilabel import read_dataset, wrap_preprocess, binarize
import json
from tqdm import tqdm
import pandas as pd

CACHE = "/scratch/project_2009199/cache"
RESULTS = "/scratch/project_2009199/register-vs-genre/register-and-genre/new_results/"
int_bs=8
threshold = 0.3 # for genre
reg_threshold=0.4

labels_all_OLD = [
    "HI",
    "ID",
    "IN",
    "IP",
    "LY",
    "MT",
    "NA",
    "OP",
    "SP",
    "av",
    "ds",
    "dtp",
    "ed",
    "en",
    "fi",
    "it",
    "lt",
    "nb",
    "ne",
    "ob",
    "ra",
    "re",
    "rs",
    "rv",
    "sr",
]
labels_all = ['MT', 'LY', 'SP', 'ID', 'NA', 'HI', 'IN', 'OP', 'IP', 'it', 'ne', 'sr', 'nb', 're', 'en', 'ra', 'dtp', 'fi', 'lt', 'rv', 'ob', 'rs', 'av', 'ds', 'ed']

register_map = {"LABEL_"+str(k):v for k,v in zip(range(len(labels_all)), labels_all)}


# ORIGINALLY LIKE THIS
#GENRE_MODEL = "/scratch/project_2009199/register-vs-genre/genre/models/xlmr-model-focal-loss2.pt"
#REGISTER_MODEL = "/scratch/project_2009199/pytorch-registerlabeling/output/en_en/xlm-roberta-large/saved_model"

# new 20.5.2024
GENRE_MODEL = "/scratch/project_2009199/register-vs-genre/register-and-genre/genre/training/models/xlmr-large-model-bce-loss1.pt"
REGISTER_MODEL = "/scratch/project_2009199/pytorch-registerlabeling/models/xlm-roberta-large/labels_all/en_en/seed_42"

#"/scratch/project_2009199/pytorch-registerlabeling/test_balance_new/en-fi-fr-sv_en-fi-fr-sv/xlm-roberta-base/checkpoints/checkpoint-50800/"

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--base_model_genre', default="xlm-roberta-large",
                    help='Base model name for genres, used for tokenizer')
    ap.add_argument('--base_model_register', default="xlm-roberta-large",
                    help='Base model name for registers, used for tokenizer')
    ap.add_argument('--genre_model', default=GENRE_MODEL,
                    help='Path to trained genre model')
    ap.add_argument('--register_model', default=REGISTER_MODEL,
                    help='Path to trained register model')
    ap.add_argument('--data_name',type=str, choices=["register_oscar", "CORE", "TurkuNLP/genre-6"],
                    required=True, help='Name of the dataset')
    ap.add_argument('--language',  type=json.loads, default=["en"], metavar='LIST-LIKE',
                    help='Language to be used from the dataset, if applicable. Give as \'["en","zh"]\' ')
    ap.add_argument('--downsample', metavar="BOOL", type=bool,
                    default=False, help='downsample to 1/20th')
    ap.add_argument('--cache', default=CACHE, metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--batch_size', metavar='INT', type=int, default=int_bs,
                    help='Batch size for predictions')
    ap.add_argument('--seed', metavar='INT', type=int,
                    default=123, help='Random seed for splitting data')
    ap.add_argument('--results', default=RESULTS, metavar='FILE',
                    help='Path to save file.')
    return ap

def split_range(x, N):
    result = []
    start = 0
    while start <= N:
        chunk = list(range(start, min(start + x, N + 1)))
        result.append(chunk)
        start += x
    return result

def create_batches(options, dataset):
    bs = options.batch_size
    L = len(dataset)
    return split_range(bs,L)
    

#def wrap_tokenizer(tokenizer):
#    def encode_dataset(d):
#        try:
#            output = tokenizer(d['text'], return_tensors='pt', truncation= True, padding = True, max_length=512)
#            return output
#        except:     #for empty text
#            output = tokenizer(" ",return_tensors='pt', truncation= True, padding = True, max_length=512)
#            return output

    return encode_dataset

def predict(model, inputs, th=0.5):
    with torch.no_grad():
        pred=model(inputs.input_ids, attention_mask=inputs.attention_mask)
    # output of the classification layer
    logits = pred.logits.cpu().detach().numpy()[0]
    # calculate sigmoid for each
    sigm = 1.0/(1.0 + np.exp(- logits))
    # make the classification, threshold = 0.5 for register, X for genre
    target = np.array([pl > th for pl in sigm]).astype(int)
    # get the classifications' indices
    target = np.where(target == 1)   # this returns tuple for some reason
    if len(target[0]) == 0:
        target = [None]   # <= this neede for easy label conversion
        return target
    return target[0].tolist()

def run(dataset, model_genre, model_register, options):
    tokenizer_g = AutoTokenizer.from_pretrained(options.base_model_genre)
    tokenizer_r = AutoTokenizer.from_pretrained(options.base_model_register)

    # this is the ideal solution, but .map() cannot return tensors
    # and tensors are needed in the prediction
    # thus, in the loop below, tokenisation is done for one element at a time
    #encoded_dataset_g = dataset.map(wrap_tokenizer(tokenizer_g)).remove_columns(["text", "label"])
    #encoded_dataset_r = dataset.map(wrap_tokenizer(tokenizer_r)).remove_columns(["text", "label"])
    # => solution: do batching? => already a function to get indices
    # tokenizer can handle multiple texts at a time, so only thing that needs to be modified is
    # the prediction logits etc calculation

    pg=[]
    pr=[]
    for d in dataset["train"]:
        if d["text"] == None:
            d["text"] = ""
        dg = tokenizer_g(d["text"],return_tensors="pt",return_special_tokens_mask=True,truncation=True).to(model_genre.device)
        dr = tokenizer_r(d["text"],return_tensors="pt",return_special_tokens_mask=True,truncation=True).to(model_register.device)
        pg.append(predict(model_genre, dg, th=threshold))
        pr.append(predict(model_register, dr, th=reg_threshold))
    
    return pg, pr



if __name__=="__main__":
    options = argparser().parse_args(sys.argv[1:])
    

    model_genre = torch.load(options.genre_model)
    model_register = AutoModelForSequenceClassification.from_pretrained(options.register_model)
    model_genre.to('cuda')
    model_register.to('cuda')
    print("Models loaded succesfully.")

    options.genre_id2label = model_genre.config.id2label
    options.register_id2label= {k:register_map[v] for k,v in model_register.config.id2label.items()}
    print(options.genre_id2label)
    print(options.register_id2label)

    print("label dictionaries succesfully extracted.")

    # load the test data
    if options.data_name=="CORE":
        # reading as pandas as HF datasets fails for some reason
        data = pd.read_csv("/scratch/project_2009199/register-vs-genre/data/CORE/test.tsv.gz", delimiter="\t", names = ["label","id", "text"], on_bad_lines='skip')
        # for predicting make this a HF dataset
        dataset = Dataset.from_pandas(data)
    if options.data_name=="register_oscar":
        dataset = datasets.load_from_disk("/scratch/project_2009199/sampling_oscar/final_reg_oscar/en.hf")
        #dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
        
    #dataset = read_dataset(options)
    #dataset = dataset.map(wrap_preprocess(options))
    #dataset, mlb_classes = binarize(dataset, options)

    

    print(dataset, flush=True)

    print("Dataset loaded. Ready for predictions.\n")
    p_g, p_r = run(dataset, model_genre, model_register, options)

    print("PREDICTION DONE")

    data = dataset["train"].to_pandas()
    print(dataset)

    # append the results to the pandas frame because it is easier to save as the same format as CORE
    data["register_prediction"] = [[options.register_id2label.get(i,"None") for i in t] for t in p_r] #p_r #[options.register_id2label[p] for p in p_r]
    data["genre_prediction"]= [[options.genre_id2label.get(i,"None") for i in t] for t in p_g]
    #dataset = dataset.add_column("register_prediction", [[options.register_id2label.get(i,"None") for i in t] for t in p_r])
    #dataset = dataset.add_column("genre_prediction", [[options.genre_id2label.get(i,"None") for i in t] for t in p_g])

    #register_results = [[options.register_id2label.get(i,"None") for i in t] for t in p_r]
    #genre_results = [[options.genre_id2label.get(i,"None") for i in t] for t in p_g]
    #dataset = dataset.map(lambda example, idx: {"register_prediction": register_results[idx], "genre_prediction": genre_results[idx]}, with_indices=True)

    #print(dataset)
    print(f'Saving to {options.results}.tsv')
    #dataset["train"].to_csv(options.results+"large_large_03_04.tsv", sep='\t')
    data.to_csv(options.results+"large_multiL-large_03_04.tsv", sep='\t')
    print("ALL DONE")

