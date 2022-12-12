import os
import random

import numpy as np
import pandas as pd
import torch
from arguments import load_args
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import trainer


def load_data(data_path):
    df = pd.read_csv(data_path)
    texts = df.text.values.tolist()
    labels = df.label.values.tolist()
    label_0 = [i for i in labels if int(i)==0]
    print('has {} label 0 in {} total label'.format(len(label_0),len(labels)))
    text_ids = []
    att_masks = []
    for text in tqdm(texts):
        text_id = tokenizer.encode(str(text), max_length=MAX_LEN, truncation=True,pad_to_max_length=True)
        att_mask = [int(token_id>0) for token_id in text_id]
        assert len(text_id) == len(att_mask)
        text_ids.append(text_id)
        att_masks.append(att_mask)
    test_size = 0.15 
    train_x, val_x, train_y, val_y = train_test_split(texts, labels, random_state=35, test_size=test_size)
    encoded_data_train = tokenizer.batch_encode_plus(train_x,
                                                     add_special_tokens=True,
                                                     return_attention_mask=True,
                                                     pad_to_max_length=True,
                                                     max_length=MAX_LEN,
                                                     truncation=True,
                                                     return_tensors='pt')
    encoded_data_val = tokenizer.batch_encode_plus(val_x,
                                                   add_special_tokens=True,
                                                   return_attention_mask=True,
                                                   pad_to_max_length=True,
                                                   max_length=MAX_LEN,
                                                   truncation=True,
                                                   return_tensors='pt')

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(train_y)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(val_y)

    train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    args = load_args()
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    seed_val = 30
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    num_labels = args.max_labels
    cached_features_file = args.data_cache
    pre_trained = args.model_pretrained

    tokenizer = AutoTokenizer.from_pretrained(pre_trained, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_pretrained,
                                                               num_labels=3,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if os.path.exists(cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size)) and \
        os.path.exists(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size)) and args.use_cache and \
        not args.debug:
        
        print("loading data train from cache...")
        train_dataloader = torch.load(cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size))
        val_dataloader = torch.load(cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size))

    else:
        if args.debug:
            data_path = args.data_path
            train_dataloader,val_dataloader = load_data(data_path)
            print('This data will not store in cache'.upper())
        else:
            data_path = args.data_path
            print("load data from {}".format(data_path))
            train_dataloader,val_dataloader = load_data(data_path)
            torch.save(train_dataloader,cached_features_file+"cache-train-{}-{}".format(args.max_len,args.batch_size))
            torch.save(val_dataloader,cached_features_file+"cache-valid-{}-{}".format(args.max_len,args.batch_size))

    trainer(train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device)