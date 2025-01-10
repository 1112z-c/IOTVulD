import logging
import os
import time
from itertools import cycle
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from model_bert import BertModel
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)
from matplotlib.colors import ListedColormap
import copy
import numpy as np
from sklearn import manifold
from sklearn.model_selection import train_test_split
import json
from transformers import (
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          T5Config, T5ForConditionalGeneration,T5Tokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
                          )
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def all_metrics(y_true, y_pred, is_training=False):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1.requires_grad = is_training

    # Matthews Correlation Coefficient (MCC)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
    # Convert to a scalar value

    # AUC (Area Under the Curve)
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())  # Convert to numpy for sklearn
    auc = torch.tensor(auc).to(y_true.device)  # Convert back to tensor on the original device

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item(), mcc.item(), auc.item()


class BERTTrainer():
    def __init__(self, args):
        self.args = args
        model_name = "./bert-base-uncased"
        MODEL_CLASSES = {
           'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),

        }
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
        config = config_class.from_pretrained(model_name)
        config.num_labels = 2
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        model =BertModel(encoder=model,config=config,tokenizer=tokenizer,args=args)
        self.best_f1 = 0
        self.metrics = {'f1': 0, 'precision': 0, 'recall': 0,'mcc': 0,'auc': 0}
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        self.model = model.to(args.device)

        self.results_data=[]
    def train(self, trainset, devset):
        train_loader = DataLoader(trainset, batch_size=self.args.batch_size_clip, shuffle=True)
        dev_loader = DataLoader(devset, batch_size=self.args.batch_size_clip, shuffle=True)
        for epoch in range(self.args.epoch_clip):
            self.train_epoch(epoch, train_loader,dev_loader)
            logging.info('Epoch %d finished' % epoch)
        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall', 'mcc', 'auc'])
        save_path = self.args.savepath + '/result_record_bert_' + self.args.dataset + 'pretraing' + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)

    def train_epoch(self, epoch, train_loader,test_loader):
        self.model.train()
        pbar = tqdm(train_loader)

        for i, data in enumerate(pbar):
            ids, label = data
            ids, label = ids.to(self.args.device), label.to(self.args.device)

            outputs,cls,_  = self.model(input_ids=ids)
            loss = self.criterion(outputs, label)
            _, predicted = torch.max(outputs.data, dim=1)

            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
            # loss.sum().item()
            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

        valid_f1 = self.eval_epoch(test_loader)

        if valid_f1 > self.best_f1:
            self.best_f1 = valid_f1
            best_model = copy.deepcopy(self.model.state_dict())
            if best_model is not None:
                self.savemodel(epoch, best_model)

    def savemodel(self, k, best_model):
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset))

        torch.save({'state_dict': best_model}, os.path.join(self.args.savepath, self.args.dataset, f'model_best_bert_ARM.pth'))
        logger.info(f'save:{k}.pth')

    def eval_epoch(self, dev_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in tqdm(dev_loader):
                ids, label = data
                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs, cls,_ = self.model(input_ids=ids)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn,mcc,auc= all_metrics(tensor_labels, tensor_preds)
            self.results_data.append([f1, precision, recall, mcc, auc])

            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f},mcc: {:.4f}, auc: {:.4f}'
                    .format(f1, precision, recall, mcc, auc))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))
        return f1



class BertClassifierTrainer(): #cross-domain Vulnerability Detection Tool
    def __init__(self, args):
        self.args = args
        model_name = "./bert-base-uncased"
        self.metrics = {'f1': 0, 'precision': 0, 'recall': 0,'mcc':0,'auc':0}
        MODEL_CLASSES = {
            'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),

        }
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
        config = config_class.from_pretrained(model_name)
        config.num_labels = 2
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        model = BertModel(encoder=model, config=config, tokenizer=tokenizer, args=args)

        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)
        self.model = model.to(args.device)

        resume_path = './Results/mlm/TEST/model_best_bert_Micro.pth'
        logger.info('loading model checkpoint from %s..' % resume_path)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        self.results_data = []

    def train(self, s_trainset, t_trainset, t_testset):
        logging.info('Star bert classifier')
        s_trainloader = DataLoader(s_trainset, batch_size=self.args.batch_size_cla, shuffle=True)
        t_trainloader = DataLoader(t_trainset, batch_size=self.args.batch_size_cla, shuffle=True)
        t_testloader = DataLoader(t_testset, batch_size=self.args.batch_size_cla, shuffle=True)
        for epoch in range(self.args.epoch_cla):
            self.train_epoch(epoch, s_trainloader, t_trainloader)
            logging.info('Epoch %d finished' % epoch)
        self.eval_epoch(t_testloader)
        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall', 'mcc', 'auc'])
        save_path = self.args.savepath + '/result_record_bert_' + self.args.dataset + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)

    def train_epoch(self, epoch, s_train_loader, t_train_loader):
        self.model.train()
        loss_domain = torch.nn.NLLLoss()
        loss_num = 0.0
        pbar = tqdm(zip(s_train_loader, cycle(t_train_loader)), total=len(s_train_loader))
        output_batches_generator = len(s_train_loader)

        for i, (s_data, t_data) in enumerate(pbar):
            p = float(i + epoch * output_batches_generator) / self.args.epoch_cla / output_batches_generator
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_ids, s_label = s_data
            s_ids, s_label = s_ids.to(self.args.device), s_label.to(self.args.device)
            s_outputs, _, domain_output = self.model(input_ids=s_ids,  alpha=alpha)

            s_loss = self.criterion(s_outputs, s_label)  # 交叉熵损失函数

            # 处理 source domain 的损失
            batch_size_s = len(s_label)
            if batch_size_s < domain_output.size(0):
                sd_label = torch.ones(domain_output.size(0)).long().to(self.args.device)
                sd_label[:batch_size_s] = s_label
            else:
                sd_label = torch.ones(batch_size_s).long().to(self.args.device)

            err_s_domain = loss_domain(domain_output, sd_label)

            # 处理 target domain 的损失
            t_ids, t_label = t_data
            t_ids, t_label = t_ids.to(self.args.device), t_label.to(self.args.device)
            t_outputs, _, domain_output = self.model(input_ids=t_ids,  alpha=alpha)

            batch_size_t = len(t_label)
            td_label = torch.ones(batch_size_t).long().to(self.args.device)  # 目标领域标签
            err_t_domain = loss_domain(domain_output, td_label)

            # 总损失
            all_loss = s_loss + err_s_domain + err_t_domain

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            loss_num += all_loss.item()

            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=all_loss.item())

    def eval_epoch(self, dev_loader):
        self.model.eval()

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in tqdm(dev_loader):
                ids, label = data

                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs, _, _ = self.model(input_ids=ids)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn, mcc, auc = all_metrics(tensor_labels, tensor_preds)
            self.results_data.append([f1, precision, recall, mcc, auc])
            logger.info(
                'target_testset -f1: {:.4f}, precision: {:.4f}, recall: {:.4f},mcc: {:.4f}, auc: {:.4f}'
                    .format(f1, precision, recall, mcc, auc))
            logger.info('target_testset -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))
