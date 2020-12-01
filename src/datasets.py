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
        
   
    def __getitem__(self, idx):
        # 데이터 형식에 따라 바꾸기
        text = self.data[idx]['text']
        sentiment = self.data[idx]['sentiment']

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
            'label': sentiment
        }

        for k, v in input.items():
            input[k] = torch.tensor(v)
        
        return input['input_ids'], input['attention_mask'], input['token_type_ids'], input['label']

    def __len__(self):
        return len(self.data)


DATASET_LIST = {
    'Classification': BaseDataset,
    'Test': PredictDataset
}