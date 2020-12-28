from torch.utils.data import Dataset 
import torch
import json
import os

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        
        # 데이터 형식에 따라 바꾸기
        data_file = f'{mode}.json'
        data_path = os.path.join(args.data_dir, data_file)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        label_path = os.path.join(args.data_dir, 'label.json')
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        self.labels2answer = {lab:idx for idx, lab in enumerate(labels)}
        self.answer2labels = {idx:lab for idx, lab in enumerate(labels)}

        for idx, elem in enumerate(self.data):
            self.data[idx]['label'] = self.labels2answer[elem['label']]
   
    def __getitem__(self, idx):
        # 데이터 형식에 따라 바꾸기
        text = self.data[idx]['text']
        label = self.data[idx]['label']

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            pad_to_max_length=True,
            truncation = True
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask'] if 'attention_mask' in encoded else None
        token_type_ids = encoded['token_type_ids'] if 'token_type_ids' in encoded else None
        
        input = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'label': label
        }

        for k, v in input.items():
            input[k] = torch.tensor(v)
        
        return input['input_ids'], input['attention_mask'], input['token_type_ids'], input['label'], text

    def __len__(self):
        return len(self.data)


DATASET_LIST = {
    'Classification': BaseDataset
}