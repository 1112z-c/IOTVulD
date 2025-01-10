from torch.utils.data import Dataset
from random import choice
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import logging
import torch
import json
import numpy as np
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer
class ContractDataSet(Dataset):
    def __init__(self, data, label):
        super(ContractDataSet, self).__init__()

        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])


def split_dataset(all_data, all_label, test_size=0.2):

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=test_size, random_state=666)

    logger.info("split dataset into %d train and %d val" % (len(train_data), len(test_data)))
    return train_data, test_data, train_label, test_label

def load_data(args):

    if args.dataset == 'FED_TEST':
        p_dataset, train_dataset_source, test_dataset_source, train_dataset_target,test_dataset_target,all_dataset_source= load_riot_data(args)

    else:
        raise ValueError('No such dataset')

    return p_dataset, train_dataset_source, test_dataset_source, train_dataset_target,test_dataset_target,all_dataset_source



def load_riot_data(args):
    all_label_source = []
    all_data_source = []
    #tokenizer = AutoTokenizer.from_pretrained("./CodeBERT/microsoft/codebert-base")
    #tokenizer = AutoTokenizer.from_pretrained("./codet5-base")
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    # 读取source-JSON数据
    vocab_size = tokenizer.vocab_size

    with open(r'./data/Micropython_data.json', 'r', encoding='utf-8') as f:
        datas_source = json.load(f)
        for contract_id, contract in datas_source.items():

            code = contract['code']
            label = contract.get('label', 0)
            inputs = tokenizer(
                code,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=args.max_length
            )['input_ids']
            code_embedding = inputs.squeeze(0)
            all_data_source.append(code_embedding)
            all_label_source.append(label)

    all_data_source = torch.stack(all_data_source)
    if (all_data_source >= vocab_size).any():
        raise ValueError("input_ids contains indices out of vocabulary range")
    all_label_source = torch.tensor(all_label_source)

    train_data_source, test_data_source, train_label_source, test_label_source\
        = train_test_split(all_data_source, all_label_source, test_size=0.8)

    smote = SMOTE(random_state=1000)
    train_data_source, train_label_source = smote.fit_resample(train_data_source.numpy().reshape(-1, args.max_length),
                                                         train_label_source.numpy())

    train_data_source = torch.tensor(train_data_source)
    train_label_source = torch.tensor(train_label_source)
    all_data_source = ContractDataSet(all_data_source,all_label_source)
    train_dataset_source = ContractDataSet(train_data_source, train_label_source)
    test_dataset_source = ContractDataSet(test_data_source, test_label_source)



    print('IOTVulCode source_dataset loaded successfully!')

    all_label_target = []
    all_data_target = []
    # 读取target-JSON数据
    with open(r'./data/ARM_data.json', 'r', encoding='utf-8') as f:
        datas_target = json.load(f)

        for contract_id, contract in datas_target.items():

            code = contract['code']
            label = contract.get('label', 0)

            inputs = tokenizer(code, padding='max_length', truncation=True, return_tensors='pt',
                               max_length=args.max_length)['input_ids']
            code_embedding = inputs.squeeze(0)
            all_data_target.append(code_embedding)
            all_label_target.append(label)

    all_data_target = torch.stack(all_data_target)
    all_label_target = torch.tensor(all_label_target)


    train_data_target, test_data_target, train_label_target, test_label_target = train_test_split(all_data_target,
                                                                                                  all_label_target,
                                                                                                  test_size=0.5)

    train_data_target = torch.tensor(train_data_target)
    train_dataset_target = ContractDataSet(train_data_target, train_label_target)
    test_dataset_target = ContractDataSet(test_data_target, test_label_target)


    print('IOTVulCode target_dataset loaded successfully!')
    return train_dataset_source, test_dataset_source, train_dataset_target,test_dataset_target, all_data_source


