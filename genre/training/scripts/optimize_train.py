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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support,  roc_auc_score
#from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EarlyStoppingCallback
import json
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import optuna

# tokenizer warning for old transformer version
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# this is to choose are we optimising multiclass or multilabel:
from importlib import import_module
# Can also be done like this:
#from train_multiclass import read_dataset, flatten, process_and_tokenize_dataset, MultiClassTrainer, compute_metrics


# HYPERPARAMS
LEARNING_RATE=1e-4
BATCH_SIZE=32
TRAIN_EPOCHS=5
WEIGHT_DECAY= 0.023716698369758392
MODEL_NAME = 'bert-base-cased'
DATASET = "TurkuNLP/genre-6"
PATIENCE=2
RESULT_FILE = '/scratch/project_2002026/amanda/genre_classification/results/results_optimisation.txt'
CACHE = "/scratch/project_2002026/amanda/cache/"
TEXT_MAX_LENGTH = 10000 # truncate texts to this lenght (in char) to make tokenization faster
# global variables
threshold = 0.3
LABELS = []
id2label = {}
label2id = {}

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mod', required=True, type=str, 
                    choices = ["train_multiclass", "train_multilabel"],
                    help='Which type of model, multilabel or multiclass, to optimize')
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--dataset', type=str, default=DATASET,
                    help='Dataset name')
    ap.add_argument('--split', type=str, default="split1",
                    help='which split to use in the dataset')
    ap.add_argument('--result_file', default = RESULT_FILE,
                    help="where to save results")
    ap.add_argument('--truncate_texts', type = bool, default=False,
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





# -------------------------------training optimisation-------------------------------#


def objective(trial: optuna.Trial):

    model = transformers.AutoModelForSequenceClassification.from_pretrained(options.model_name, 
                                                                            num_labels=len(LABELS), 
                                                                            cache_dir=CACHE
                                                                           )
    
    data_collator = transformers.DataCollatorWithPadding(tokenizer,padding='max_length')

    trainer_args = TrainingArguments(
        output_dir='checkpoints',
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=trial.suggest_float('learning_rate',low = 1e-6,high=1e-4, log=True),
        #weight_decay =trial.suggest_float('weight_decay', low=1e-4, high=5e-1, log=True),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=trial.suggest_int('epochs', low=2, high=8)
    )

    
    trainer = module.MultiTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=module.compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer
    )

    
    # Results = how well the training is going wrt loss
    results = trainer.train()
    
    if options.save_model is not None:
        parsed_learningrate = str(trainer_args.learning_rate).split("e")[0][:5].replace(".","_")+"e"+str(trainer_args.learning_rate).split("e")[1]
        full_model_name = options.save_model+"_"+parsed_learningrate+"_"+str(int(trainer_args.num_train_epochs))+".pt"
        torch.save(trainer.model, full_model_name)#"models/multilabel_model3_fifrsv.pt")


    return results.training_loss


# ---------------------------------------main---------------------------------------#

if __name__=="__main__":
    print("---------------------------------------------")
    
    options = argparser().parse_args(sys.argv[1:])

    # import what needs to be optimized
    module = import_module(options.mod)

    # reading and preprocessing 
    dataset = module.read_dataset(options)

    LABELS = np.unique(module.flatten(dataset["train"]["label"]))
    print(f'LABELS found in data: {LABELS}')

    # tokenisation and label encoding
    dataset, tokenizer = module.process_and_tokenize_dataset(dataset, options)
    print("Preprocessing done")

    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 7)
    
    print(study.best_params)
    
    with open(options.result_file, 'w') as f:
        f.write(json.dumps(study.best_params))
    f.close()
    
    print("---------------------------------------------")
