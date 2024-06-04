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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from transformers import EarlyStoppingCallback
import json
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# tokenizer warning for old transformer version
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# HYPERPARAMS (optimized with optimize_train.py EXCEPT fiction!)
"""
LEARNING_RATE = {"split1":3.224016966388526e-05, 
                 "split2":4.022628330511456e-05, 
                 "split3":3.0665891577600484e-05, 
                 "fiction":3e-5}
WEIGHT_DECAY = {"split1":0.00035098201181716645, 
                "split2":0.030510211788901346, 
                "split3":0.004544484499043511, 
                "fiction":3e-3}
TRAIN_EPOCHS = {"split1":8, 
                "split2":5, 
                "split3":8, 
                "fiction":3}
"""


# optimisized with random page
LEARNING_RATE = {"split1": 3.5e-05, 
                 "split2":1.5e-05, 
                 "split3":4e-05, 
                 "fiction":3e-5}
WEIGHT_DECAY = {"split1":1e-2, 
                "split2":9e-2, 
                "split3": 1e-2, 
                "fiction":3e-2}
TRAIN_EPOCHS = {"split1":5, 
                "split2":5, 
                "split3":5, 
                "fiction":3}




BATCH_SIZE=32
MODEL_NAME = 'bert-base-cased'
DATASET = "TurkuNLP/genre-6"
RESULT_FILE = '/scratch/project_2009199/genre/results/results_multiclass.txt'
CACHE = "/scratch/project_2009199/cache/"
TEXT_MAX_LENGTH = 10000 # truncate texts to this lenght (in char) to make tokenization faster

# Global vars that will be modified after reading arguments:

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
    ap.add_argument('--checkpoints', default='checkpoints', metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Save model to file, if given')
    ap.add_argument('--seed', default=123,
                    help='seed for random behaviour (shuffle)')
    ap.add_argument('--random_excerpt', default=True,
                    help='selects random 10000 characters ~ 6 pages of book instead of the beginning') 
    return ap



#-------------------------------------processing-------------------------------------#

def read_dataset_with_path(options):
    """ Read the data from 3 .parquet files"""
    print("Reading data")
    dataset = load_dataset("parquet", data_files={'train': options.path+'train.parquet',
                                                  #'test': path+'test.parquet',
                                                  'dev': options.path+'dev.parquet'},
                           cache_dir=CACHE)
    labels = np.unique(flatten(dataset[list(dataset.keys())[0]]["label"]))   #dataset.unique("label")
    global label2id, id2label
    label2id = {k:v for k,v in zip(labels, range(len(labels)))}
    id2label = {v:k for k,v in zip(labels, range(len(labels)))}
    print(id2label)
    dataset = dataset.remove_columns("categories")
    return dataset.shuffle(options.seed)


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

    labels = np.unique(flatten(dataset[list(dataset.keys())[0]]["label"]))   #dataset.unique("label")
    global label2id, id2label
    label2id = {k:v for k,v in zip(labels, range(len(labels)))}
    id2label = {v:k for k,v in zip(labels, range(len(labels)))}
    return dataset.shuffle(options.seed)


def flatten(matrix):
    """ Flattening that works for both nested lists and unnested lists """
    return [item for row in matrix for item in (row if isinstance(row, list) else [row])]
    
def label_encoding(d):
    """ Encode label to a number. The extracted "label" encoded and saved to field "labels" as BERT needs a plural ;) 
    For multiclass, take the first label only. """
    if type(d["label"]) == list:
        d["labels"] = label2id[d["label"][0]] 
    else:
        d["labels"] = label2id[d["label"]]
    return d


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


def process_and_tokenize_dataset(dataset, options):
    
    print("Encoding the labels")
    # labels to numbers
    dataset= dataset.map(label_encoding)
    print(f'Tokenisation, truncation = {options.truncate_texts}')
    # tokenization
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name)
    dataset = dataset.map(wrap_tokenizer(tokenizer, options))
    
    # remove fields that confuse Bert... encoded label saved in "labels", not label
    dataset = dataset.remove_columns(["text", "label"])
    
    return dataset, tokenizer

#--------------------------------------metrics--------------------------------------#

def compute_metrics(pred):
    """ Return micro & macro f1 and accuracy for given predictions"""
    predictions = pred.predictions.argmax(axis=1)
    y_pred = predictions
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


#-------------------------------------Trainer-------------------------------------#

# define loss for multiclass
class MultiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss


#------------------------------------training------------------------------------#
    

def train_multiclass(dataset, tokenizer, options):
    
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
                                                                            num_labels=len(id2label), 
                                                                            cache_dir=CACHE,
                                                                            id2label=id2label, label2id=label2id)
    trainer = MultiTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]
    )
    
    
    print("Ready to train")
    trainer.train()
    
    print("Evaluating with dev set...")
    predictions, true_labels, eval_results =trainer.predict(dataset["dev"])
    
    # calculate label for all predictions (as argmax) 
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    pred_labels = torch.argmax(probs, dim=1)
    
    # create report and confusion matrix for visualisation
    clsfr = classification_report(true_labels, pred_labels, target_names=id2label.values())
    print(clsfr)
    cfm = confusion_matrix(true_labels, pred_labels)
    #print(cfm)
        
    # write results
    with open(options.result_file, 'w') as f:
        f.write(json.dumps(eval_results))
        f.write(str(clsfr))
        f.write(str(cfm))
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
    

#----------------------------------------main---------------------------------------#



if __name__=="__main__":
    print("---------------------------------------------")
    
    options = argparser().parse_args(sys.argv[1:])
    options = parse_further(options)
    print("Training multiclass classifier.")
    print(options)
    options.path = "/scratch/project_2002026/veronika/genre-stuff/splits/five_cats_test2/"
    
    # reading and preprocessing 
    dataset = read_dataset_with_path(options)


    # tokenisation and label encoding
    dataset, tokenizer = process_and_tokenize_dataset(dataset, options)
    print("Preprocessing done")

    
    # training and evaluation
    train_multiclass(dataset, tokenizer, options)
    
    print("---------------------------------------------")
