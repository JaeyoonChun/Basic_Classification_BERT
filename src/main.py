from trainer import Trainer
from utils import init_logger, set_seeds, load_tokenizer

import os
import argparse
from attrdict import AttrDict
import json

def main(opt):
    with open('config.json', 'r', encoding='utf-8') as f:
        args = AttrDict(json.load(f))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    init_logger()
    set_seeds()
    tokenizer = load_tokenizer(args)
    
    if opt.train:
        trainer = Trainer(args, tokenizer)
        trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Whether to run training.")

    args = parser.parse_args()

    main(args)

