import logging
import random
import numpy as np
import torch
import argparse
from data_utils_bert import load_data
from train_bert import  BertClassifierTrainer,BERTTrainer
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', type=str, default='ARM_FreeRTOS',
                        choices=['FreeRTOS_ARM', 'FreeRTOS_Micro', 'ARM_FreeRTOS', 'ARM_Micro', 'Micro_FreeRTOS',
                                 'Micro_ARM', 'TEST','Micro_RIOT','Micro_OPEN'])

    parser.add_argument('--epoch_clip', type=int, default=50)
    parser.add_argument('--batch_size_clip', type=int, default=16)
    parser.add_argument('--lr_clip', type=float, default=1e-5)
    parser.add_argument('--save_epoch', type=int, default=1)


    parser.add_argument('--epoch_cla', type=int, default=10)
    parser.add_argument('--batch_size_cla', type=int, default=16)
    parser.add_argument('--lr_2', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--savepath', type=str, default='./Results/mlm')


    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    set_seed(args.seed)

    # get data
    train_dataset_source, test_dataset_source, train_dataset_target, test_dataset_target,all_dataset_source = load_data(args)
    trainer = BERTTrainer(args)
    trainer.train(train_dataset_source, test_dataset_source)

    cla_trainer = BertClassifierTrainer(args)
    cla_trainer.train(train_dataset_source, train_dataset_target, test_dataset_target)
