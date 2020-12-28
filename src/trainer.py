import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.modeling_bert import BertLayerNorm

from datasets import DATASET_LIST
from utils import compute_metrics

import os
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, tokenizer):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(self.args.bert_type,
                                                num_labels=self.args.num_labels, 
                                                finetuning_task=args.task)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.bert_type, config=self.config).to(self.device)

        if self.args.init_pooler:
            # pooling layer
            encoder_temp = getattr(self.model, "bert")
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True

        if self.args.init_layers:
            assert self.args.init_nums != 0
            encoder_temp = getattr(self.model, "bert")
            for layer in encoder_temp.encoder.layer[-self.args.init_nums:]:
                for module in layer.modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    elif isinstance(module, BertLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, torch.nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()

    def train(self):
        train_dataset = DATASET_LIST[self.args.model_mode](self.args, self.tokenizer, "train")
        valid_dataset = DATASET_LIST[self.args.model_mode](self.args, self.tokenizer, "dev")
      
        train_dataloader = DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.args.train_batch_size)
        valid_dataloader = DataLoader(dataset=valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=self.args.eval_batch_size)
      
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # save hyperparameters
        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        with open(f'{self.args.save_model_dir}/hyperparameter', 'w', encoding='utf-8') as f:
            f.write(f"  Num train data = {len(train_dataset)}\n")
            f.write(f"  Num valid data = {len(valid_dataset)}\n")
            f.write(f"  Total train batch size = {self.args.train_batch_size}\n")
            f.write(f"  Gradient Accumulation steps ={self.args.gradient_accumulation_steps}\n")
            f.write(f"  Total optimization steps = {t_total}\n")
            f.write(f"  Logging steps = {self.args.logging_steps}\n")
            f.write(f"  saving steps = {self.args.save_steps}\n")
            f.write(f"  learning rate = {self.args.learning_rate}\n")
            f.write(f"  weight decay = {self.args.weight_decay}\n")

        self.model.zero_grad()
        best_valid_loss = float('inf')
        global_step = 0
        training_stats = []

        for epoch_idx in range(int(self.args.num_train_epochs)):            
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            epoch_train_loss, epoch_valid_loss = 0, 0
            epoch_valid_accuracy, valid_cnt = 0, 0

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                    }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                epoch_train_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    
                    if self.args.logging_steps > 0 and global_step  % self.args.logging_steps == 0:
                        valid_loss, valid_accuracy = self.evaluate(valid_dataloader, "valid")
                        logger.info(f"  global steps = {global_step}")
                        logger.info(f'  learning rate = {scheduler.get_last_lr()[0]}')
                        epoch_valid_loss += valid_loss
                        epoch_valid_accuracy += valid_accuracy
                        valid_cnt += 1

                        training_stats.append(
                                {
                                    'epoch': epoch_idx + 1,
                                    'training_loss': epoch_train_loss / (step + 1),
                                    'valid_loss': valid_loss,
                                    'valid_accuracy': valid_accuracy,
                                    'steps': global_step,
                                    'lr': scheduler.get_last_lr()[0]
                                }
                            )
                    
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            self.save_model(optimizer, global_step, best=True)
                        else:
                            self.save_model(optimizer, global_step)
                                            
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            
            if 0 < self.args.max_steps < global_step:
                df_stats = pd.DataFrame(data=training_stats, )
                df_stats = df_stats.set_index('epoch')
                df_stats.to_csv(f'{self.args.save_model_dir}/stats.csv', sep='\t', index=True)
                break


    def evaluate(self, eval_dataloader, mode):
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]}
                
                outputs = self.model(**inputs)

                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results['loss'], results['acc']

    def save_model(self, optimizer, steps, best=False):
        if best: ckpt = 'ckpt-best'
        else: ckpt =f'ckpt-{steps}'
        ckpt = os.path.join(self.args.save_model_dir, ckpt)

        if not os.path.exists(self.args.save_model_dir):
            os.makedirs(self.args.save_model_dir)
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(ckpt)
        
        torch.save(self.args, os.path.join(ckpt, 'training_args.bin'))
        logger.info(f"Saving model checkpoint to {ckpt}")