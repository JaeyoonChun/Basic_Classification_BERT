import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets_pre import DATASET_LIST
from transformers import AutoModelForSequenceClassification,  AutoTokenizer
from collections import defaultdict
from utils import init_logger, binary_accuracy, load_tokenizer, compute_metrics
from sklearn.metrics import confusion_matrix
import json
from attrdict import AttrDict
import sys
import glob
import time

logger = logging.getLogger(__name__)


def predict(args, model, tokenizer, device, test_dataloader):    
    preds = None
    out_label_ids = None
    nb_eval_steps = 0
    model.eval()
    texts = []
    for batch in tqdm(test_dataloader, desc="Predicting"):
        
        texts.append(batch[-1])
        batch = batch[:-1]

        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": None
                }
            label = batch[3]                    
            outputs = model(**inputs)
            logits = outputs[0]
        
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label.detach().cpu().numpy(), axis=0)
    
    return preds, out_label_ids, texts


def main():
    with open('config_pre.json', 'r', encoding='utf-8') as f:
        args = AttrDict(json.load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer(args)

    model = AutoModelForSequenceClassification.from_pretrained(args.save_model_dir).to(device)

    test_dataset = DATASET_LIST['Test'](args, tokenizer, "test")
    test_dataloader = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)
    
    all_preds, all_out_label_ids, texts = predict(args, model, tokenizer, device, test_dataloader)
    all_preds_argmax = np.argmax(all_preds, axis=1)
    
    result = [{"id": idx, "text": t[0], "label":an} for idx, (t, an) in enumerate(zip(texts, all_preds_argmax))]
    result = {'annotations': result}
    with open(args.result_dir, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent='\t')


if __name__ == '__main__':    
    main()
