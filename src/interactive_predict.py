import torch
import torch.nn as nn
from utils import load_tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BertModel
import numpy as np
import time
import logging
from attrdict import AttrDict
import json

logger = logging.getLogger(__name__)

def convert_example_to_features(sentence, tokenizer, args):
    max_seq_length = args.max_seq_length
    tokens_a = tokenizer.tokenize(sentence)

    _truncate_seq_pair(tokens_a,"", max_seq_length - 3)

    tokens = []
    token_type_ids = []
    tokens.append("[CLS]")
    token_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        token_type_ids.append(0)
    tokens.append("[SEP]")
    token_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        token_type_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    tensors = (torch.tensor(input_ids),
                torch.tensor(input_mask),
                torch.tensor(token_type_ids)
            )
    return tensors

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def interactive_predict():
    with open('config_pre.json', 'r', encoding='utf-8') as f:
        args = AttrDict(json.load(f))
        
    logger.info("***** Loading... *****")
    start = time.time()
    
    tokenizer = load_tokenizer(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.bert_type,
                                                    num_labels=2, 
                                                    finetuning_task=args.task,
                                                    id2label={str(i): label for i, label in enumerate([0, 1])},
                                                    label2id={label: i for i, label in enumerate([0, 1])})
    model = AutoModelForSequenceClassification.from_pretrained(args.test_model_dir, config=config).to(device)
    end = time.time()
    logger.info(f"***** Model Loaded: It takes {end-start:.2f} sec *****")

    model.eval()
    while True:
        sentence = input('\n문장을 입력하세요: ')
        tensors = convert_example_to_features(sentence, tokenizer)
        batch = tuple(t.to(device) for t in tensors)
     
        with torch.no_grad():     
            inputs = {"input_ids": batch[0].unsqueeze(0),
                        "attention_mask": batch[1].unsqueeze(0),
                        "token_type_ids": batch[2].unsqueeze(0)
                    }
            outputs = model(**inputs)
            logits = outputs[0]
       
        prob = np.max(torch.sigmoid(logits).detach().cpu().numpy())
        logits = logits.detach().cpu().numpy()
        if np.argmax(logits) == 1:
            print(f'{prob*100:.0f}% 확률로 긍정 문장입니다.')
        else:
            print(f'{prob*100:.0f}% 확률로 부정 문장입니다.')

if __name__ == '__main__':
    interactive_predict()