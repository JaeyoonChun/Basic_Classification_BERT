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

    checkpoints = sorted([dir for dir in glob.glob(f'{args.save_model_dir}/*') if os.path.isdir(dir)])
    if not args.eval_all_ckpts: checkpoints = checkpoints[-1]

    results = {}
    eval_preds, eval_labels = [], []
    for ckpt in checkpoints:
        steps = ckpt.split('-')[-1]
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)

        test_dataset = DATASET_LIST['Test'](args, tokenizer, "test")
        test_dataloader = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)
        
        all_preds, all_out_label_ids, texts = predict(args, model, tokenizer, device, test_dataloader)
        all_preds_argmax = np.argmax(all_preds, axis=1)
        
        eval_preds.append(all_preds_argmax)
        eval_labels.append(all_out_label_ids)
        results[steps] = compute_metrics(all_preds_argmax, all_out_label_ids)

        result = [{"id": idx, "text": t[0], "label":an} for idx, (t, an) in enumerate(zip(texts, all_preds_argmax))]
        result = {'annotations': result}
        with open(os.path.join(ckpt, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent='\t')
    
    with open(os.path.join(args.save_model_dir, 'eval_results.txt'), 'w', encoding='utf-8') as f:    
        for idx, key in enumerate(sorted(results.keys())):
            print(f"{key}: {str(results[key]['acc'])}")
            print(confusion_matrix(eval_labels[idx], eval_preds[idx], [1, 0, 2, 3, 4]).tolist())
            print()
            f.write(f"{key}: {str(results[key]['acc'])}\n")
            f.write(f"{confusion_matrix(eval_labels[idx], eval_preds[idx], [1, 0, 2, 3, 4]).tolist()}\n\n")

if __name__ == '__main__':    
    main()
