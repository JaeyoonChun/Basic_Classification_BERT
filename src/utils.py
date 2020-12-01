import os
import logging
import logging.config
import torch
import random
from transformers import AutoTokenizer
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
   
def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.bert_type)

def set_seeds():
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds)).argmax(axis=1)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return rounded_preds, acc

def format_time(end, start):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }
