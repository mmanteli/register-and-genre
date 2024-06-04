import transformers
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import disable_caching
disable_caching()
import torch
import logging
import sys
import numpy as np
logging.disable(logging.INFO)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support,roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EarlyStoppingCallback
import json
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# tokenizer warning for old transformer version
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
# HYPERPARAMS (optimized with optimize_train.py EXCEPT fiction!)
LEARNING_RATE = {"split1":0.00013677171574265318, 
                 "split2":2.265033939850274e-05, 
                 "split3":2.959309107485233e-05, 
                 "fiction":3e-5}
WEIGHT_DECAY = {"split1":0.00047271670493588205, 
                "split2":0.008379426319783138, 
                "split3":0.18351830107608347, 
                "fiction":3e-3}
TRAIN_EPOCHS = {"split1":2,
                "split2":4, 
                "split3":8, 
                "fiction":3}
"""

# optimized with random pages
LEARNING_RATE = {"split1":9.441733992881222e-05, 
                 "split2":4.392611968727189e-05, 
                 "split3":5.429645063759785e-05, 
                 "fiction":3e-5}
WEIGHT_DECAY = {"split1":0.08429295425351, 
                "split2":0.12342706195552643, 
                "split3":0.29120592860763955, 
                "fiction":3e-3}
TRAIN_EPOCHS = {"split1":8,
                "split2":8, 
                "split3":3, 
                "fiction":3}

threshold = 0.3
BATCH_SIZE=16
MODEL_NAME = 'bert-base-cased'
DATASET = "TurkuNLP/genre-6"
RESULT_FILE = '/scratch/project_2009199/genre/results/results_multilabel.txt'
CACHE =  "/scratch/project_2009199/cache/"
TEXT_MAX_LENGTH = 10000 # truncate texts to this lenght (in char) to make tokenization faster

# Global vars that will be modified after reading arguments:
LABELS = []
id2label = {}
label2id = {}


# ABOUT CATEGORIES
# Each book has 1-3 genres (categories)
# the dataset in huggingface provides 3 readymade splits
# here is mapping for all categories that have more than 200 instances
all_cats_200plus = {"LIT" : "Literature & Fiction",
                    "POL" : "Politics & Social Sciences",
                    "ENG" : "Engineering & Transportation",
                    "REL" : "Religion",
                    "SFI" : "Science Fiction",
                    "GUI" : "References & Guides & Self-Help ", #needs a space
                    "HIS" : "History",
                    "DIY" : "Hardware & DIY & Home",
                    "ART" : "Arts & Photography",
                    "FAN" : "Fantasy",
                    "EDU" : "Education & Teaching",
                    "BIO" : "Biology & Nature & Biological Sciences",
                    "CHI" : "Children's Books",
                    "SMA" : "Science & Math",
                    "YOU" : "Teen & Young Adult & College",
                    "COS" : "Computer Science",
                    "MED" : "Medicine & Health Sciences",
                    "FIT" : "Health, Fitness & Dieting",
                    "ACT" : "Activities, Crafts & Games",
                    "MEM" : "Biographies & Memoirs",
                    "ENT" : "Humor & Entertainment",
                    "EAT" : "Cookbooks, Food & Wine",
                    "COM" : "Comics & Graphic Novels",
                    "MIL" : "Military",
                    "GEO" : "Geography & Cultures",
                    "BUS" : "Business & Money"
                   }



def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--dataset', type=str, default=DATASET,
                    help='Dataset name')
    ap.add_argument('--split', type=str, default="split1",
                    help='which split to use in the dataset')
    ap.add_argument('--result_file', default = RESULT_FILE,
                    help="where to save results")
    ap.add_argument('--truncate_texts', type = bool, default=True,
                    help='Truncate texts for before tokenisation to make it faster')
    ap.add_argument('--batch_size', metavar='INT', type=int,default=BATCH_SIZE,
                    help='Batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs, default value optimized')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate, default value optimized')
    ap.add_argument('--wd', '--weight_decay', metavar='FLOAT', type=float,
                    default=WEIGHT_DECAY, help='Weight decay, default value optimized')
    #ap.add_argument('--tr', '--threshold', metavar='FLOAT', type=float,
    #                default=THRESHOLD, help='thre')
    ap.add_argument('--checkpoints', default='checkpoints', metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Save model to file, if given')
    ap.add_argument('--seed', default=123,
                    help='seed for random behaviour (shuffle)')
    ap.add_argument('--random_excerpt', default=True,
                    help='selects random 10000 characters ~ 6 pages of book instead of the beginning') 
    return ap



# -------------------------------------processing-------------------------------------#

def map_label_to_list(d):
    if type(d["label"])==str:
        d["label"] = [d["label"]]
    return d


def read_dataset(options):
    """ Read the data """
    print("Reading data")
    dataset = load_dataset(options.dataset, cache_dir=CACHE)
    #dataset = load_dataset("json", data_files={
    #                                "train": "/scratch/project_2002026/amanda/genre_classification/hf-dataset-gen/upload/train.jsonl.gz",
    #                                "test":"/scratch/project_2002026/amanda/genre_classification/hf-dataset-gen/upload/test.jsonl.gz",
    #                                "dev": "/scratch/project_2002026/amanda/genre_classification/hf-dataset-gen/upload/validation.jsonl.gz"
    #                                },
    #                       cache_dir=CACHE
    #                      )

    if options.split in dataset[list(dataset.keys())[0]].column_names:
        dataset = dataset.select_columns(["text",str(options.split)])
        dataset = dataset.filter(lambda example: example[str(options.split)]!=[])
        dataset = dataset.rename_column(options.split, "label")
        # fiction split currently has the labels in different format (as string, not as list of strings)
        # this can be removed when the hf-dataset is updated.
        if options.split=="fiction":
            dataset = dataset.map(map_label_to_list)
    else:
        # define your own label mapping here
        raise Exception("Not an appropriate split. Give an existing split, or define your own function here")

    global LABELS
    LABELS = np.unique(flatten(dataset[list(dataset.keys())[0]]["label"]))   #dataset.unique("label")
    return dataset.shuffle(options.seed)

"""
def categories_to_labels(d):
    # this is a function that could be modified if you want to make other splits yourself

    # separate categories
    categories = d['categories'].split(";")
    # choose the ones that are wanted by the user
    accepted = filter(lambda x: x in LABELS, categories)
    try:
        d['label'] = [i for i in accepted] #a suitable category was found, choosing first -> Read TODO
    except:
        d['label'] = [] # if the book does not contain any wanted genres, return empty and remove in the next step
    return d
"""

def flatten(matrix):
    """ Flattening that works for both nested lists and unnested lists """
    return [item for row in matrix for item in (row if isinstance(row, list) else [row])]
    


def wrap_tokenizer(tokenizer, options):
    """Wrapping to allow dataset.map() to have the tokenizer as a parameter"""
    np.random.seed(options.seed)
    def encode_dataset(d):
        """
        Tokenize the sentences.
        """
        if not options.truncate_texts:
            output = tokenizer(d['text'], truncation= True, padding = True, max_length=512)
            return output
        else: # Truncation makes this quite a bit faster, use esp. in testing
            if options.random_excerpt and len(d["text"])>TEXT_MAX_LENGTH:
                a = np.random.randint(0, len(d["text"])-TEXT_MAX_LENGTH)
                output = tokenizer(d["text"][a:a+TEXT_MAX_LENGTH], truncation= True, padding = True, max_length=512)
            else:
                output = tokenizer(d['text'][0:TEXT_MAX_LENGTH], truncation= True, padding = True, max_length=512)
            return output 
    return encode_dataset


def binarize(dataset):
    """ Binarize the labels of the data. Fitting based on the LABELS extracted during reading data"""
    mlb = MultiLabelBinarizer()
    mlb.fit([LABELS])
    dataset = dataset.map(lambda line: {'labels': mlb.transform([line['label']])[0]})
    global label2id
    global id2label
    label2id = dict(zip(mlb.classes_, [i for i in range(0, len(mlb.classes_))]))
    id2label = dict(zip([i for i in range(0, len(mlb.classes_))], mlb.classes_))
    return dataset, mlb.classes_


def process_and_tokenize_dataset(dataset, options):
    """ Change labels to binary vectors and tokenize texts """

    print("Binarizing labels")
    # labels vectors
    dataset, mlb_keys = binarize(dataset)
    print(f'Tokenisation, truncation = {options.truncate_texts}')
    # tokenizations
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name)
    dataset = dataset.map(wrap_tokenizer(tokenizer, options))
    
    # remove fields that confuse Bert... binarized label saved in "labels"
    dataset = dataset.remove_columns(["text", "label"])
    
    return dataset, tokenizer

# --------------------------------------metrics--------------------------------------#


"""
def compute_metrics(pred):
    
    #predictions = pred.predictions.argmax(axis=1)   # for multiclass
    probs = pred.predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = pred.label_ids
    
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'accuracy': accuracy
    }
    return metrics
"""

def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average, # user-chosen or optimized threshold
               'f1_th05': f1_score(y_true=y_true, y_pred=y_th05, average='micro'), # report also f1-score with threshold 0.5
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    #if threshold == None:
    #    best_f1_th = optimize_threshold(preds, p.label_ids)
    #    threshold = best_f1_th
    #    print("Best threshold:", threshold)
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids,
        threshold=threshold)
    return result

# -------------------------------------Trainer-------------------------------------#

# define loss for multilabel

class MultiTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        #if options.class_weights == True:
        #    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
        #else:
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, len(LABELS)))    # testaappa tää
        return (loss, outputs) if return_outputs else loss


# ------------------------------------training------------------------------------#


def train_multilabel(dataset, tokenizer, options):
    
    data_collator = transformers.DataCollatorWithPadding(tokenizer,padding='max_length')

    trainer_args = TrainingArguments(
        output_dir=options.checkpoints,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        load_best_model_at_end=True,
        eval_steps=200,
        learning_rate= options.lr,
        weight_decay=options.wd,
        per_device_train_batch_size=options.batch_size,
        per_device_eval_batch_size=options.batch_size,
        num_train_epochs=options.epochs
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(options.model_name, 
                                                                            num_labels=len(LABELS), 
                                                                            cache_dir=CACHE,
                                                                            id2label=id2label, label2id=label2id)
    trainer = MultiTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer
    )
    
    
    print("Ready to train")
    trainer.train()
    
    print("Evaluating with dev set...")
    predictions, true_labels, eval_results =trainer.predict(dataset["dev"])
    
    # calculate label for all predictions 
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # with threshold (multiple labels)
    pred_labels = np.zeros(probs.shape)
    pred_labels[np.where(probs >= threshold)] = 1
    
    # with argmax (one label)
    #pred_labels = torch.argmax(probs, dim=1)
    
    # create report and confusion matrix for visualisation
    clsfr = classification_report(true_labels, pred_labels, target_names=label2id.keys())
    print(clsfr)
    #cfm = confusion_matrix(true_labels, pred_labels)
    #print(cfm)
        
    # write results
    with open(options.result_file, 'w') as f:
        f.write(json.dumps(eval_results))
        f.write(str(clsfr))
        #f.write(str(cfm))
    f.close()
    
    if options.save_model is not None:
        torch.save(trainer.model, options.save_model)


#-------------------------------arguments for splits------------------------------#

def parse_further(options):
    if options.lr==None or type(options.lr)==dict:
        options.lr=LEARNING_RATE[options.split]
    if options.wd==None or type(options.wd)==dict:
        options.wd=WEIGHT_DECAY[options.split]
    if options.epochs==None or type(options.epochs)==dict:
        options.epochs=TRAIN_EPOCHS[options.split]
    return options
    

# ---------------------------------------main---------------------------------------#

if __name__=="__main__":
    print("---------------------------------------------")
    
    options = argparser().parse_args(sys.argv[1:])
    options = parse_further(options)
    print("Training multilabel classifier.")
    print(options)
    
    
    # reading and preprocessing 
    dataset = read_dataset(options)
    
    # tokenisation and label encoding
    dataset, tokenizer = process_and_tokenize_dataset(dataset, options)
    print("Preprocessing done")
    
    
    # training and evaluation
    train_multilabel(dataset, tokenizer, options)
    
    print("---------------------------------------------")
