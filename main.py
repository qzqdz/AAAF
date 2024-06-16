# -*- encoding:utf-8 -*-
import math
import os
import pprint
import random
import re
import sys
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
import ast

from tqdm import tqdm

import util_loss
import pickle
from lion_pytorch import Lion
# from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, \
#     AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModelForCausalLM


from peft import get_peft_model, LoraConfig, TaskType



import torch.nn as nn
import argparse
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import sklearn
# import pkuseg
from torch.autograd import Variable
from knowledgegraph import KnowledgeGraph, KnowledgeSentence

import chardet
def loss_choice(loss_func_name,class_freq,train_num,model_config):

    gamma = 2
    if model_config:
        focal={}
        logit_reg={}
        map_param={}
        CB_loss={}
        if model_config['focal']:
            focal = dict(focal=True, alpha=model_config['alpha'], gamma=gamma)
        if model_config['logit_reg']:
            logit_reg = dict(init_bias=model_config['init_bias'], neg_scale=model_config['neg_scale'])
        if model_config['map_param']:
            map_param = dict(alpha=model_config['map_alpha'], beta=model_config['map_beta'], gamma=model_config['map_gamma'])
        if model_config['CB_loss']:
            CB_loss = dict(alpha=model_config['CB_loss_alpha'], CB_mode='by_class')

        loss_fct = util_loss.ResampleLoss(reweight_func=model_config['reweight_func'], loss_weight=model_config['loss_weight'],
                                              focal=focal,
                                              logit_reg=logit_reg,
                                              map_param=map_param,
                                              CB_loss=CB_loss,
                                              class_freq=class_freq, train_num=train_num
                                          )
    else:
        if loss_func_name == 'FL':
            loss_fct = util_loss.ResampleLoss(reweight_func=None, loss_weight=1.0,
                                              focal=dict(focal=True, alpha=1.0, gamma=gamma),
                                              logit_reg=dict(),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'CBloss':
            loss_fct = util_loss.ResampleLoss(reweight_func='CB', loss_weight=5.0,
                                              focal=dict(focal=True, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(),
                                              CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'R-BCE-Focal':  # R-FL
            loss_fct = util_loss.ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                              focal=dict(focal=True, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(),
                                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'NTR-Focal':  # NTR-FL
            loss_fct = util_loss.ResampleLoss(reweight_func=None, loss_weight=0.5,
                                              focal=dict(focal=True, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'DBloss-noFocal':  # DB-0FL
            loss_fct = util_loss.ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                                              focal=dict(focal=False, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'CBloss-ntr':  # CB-NTR
            loss_fct = util_loss.ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                              focal=dict(focal=True, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                              CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                              class_freq=class_freq, train_num=train_num)

        elif loss_func_name == 'DBloss':  # DB
            loss_fct = util_loss.ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                              focal=dict(focal=True, alpha=0.5, gamma=gamma),
                                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                              class_freq=class_freq, train_num=train_num)


        else:
            loss_fct = BCEWithLogitsLoss()
    return loss_fct




class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input_token(self, sentences, max_seq_len):
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        return result


    # 句子p get_input1
    def get_input1(self, sentences, knowledges, max_seq_len=128, num=1, target=None):
        for i in range(len(sentences)):
            if type(sentences[i]) == float:
                sentences[i] = ''

        if num>0:
            # Concatenate knowledges and sentences
            sentences = [f"{';'.join(knowledges[i][:min(num, len(knowledges[i]))])}[SEP]{sentence}"
                         for i, sentence in enumerate(sentences)]

        result = self.get_input_token(sentences, max_seq_len)



        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        # print('--------')
        # # print('here is the joint！')
        # print(sentences[0])
        # print(knowledges[0])
        # # print(result)
        # print('--------')

        return seqs, seq_masks, seq_segments

    # 句子p get_input1 后置拼接 get_input1_v1
    def get_input1_v1(self, sentences, knowledges, max_seq_len=128, num=1, target=None):
        for i in range(len(sentences)):
            if type(sentences[i]) == float:
                sentences[i] = ''

        if num > 0:
            # 有target
            # Concatenate knowledges and sentences
            sentences = [f"{sentence}[SEP]{target}.{';'.join(knowledgeggs[i][:min(num, len(knowledges[i]))])}" for
                         i, sentence
                         in enumerate(sentences)]

            # 无target
            # Concatenate knowledges and sentences
            # sentences = [f"{sentence}[SEP]{';'.join(knowledges[i][:min(num, len(knowledges[i]))])}" for
            #              i, sentence
            #              in enumerate(sentences)]



        result = self.get_input_token(sentences, max_seq_len)

        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        # print('--------')
        # # print('here is the joint！')
        # print(sentences[0])
        # print(knowledges[0])
        # # print(result)
        # print('--------')

        return seqs, seq_masks, seq_segments


    # 只有target get_input1_v2,
    def get_input(self, sentences, knowledges, max_seq_len=128, num=1, target=None):
        for i in range(len(sentences)):
            if type(sentences[i]) == float:
                sentences[i] = ''

        if num > 0:
            # 有target
            # Concatenate knowledges and sentences
            sentences = [f"{sentence}[SEP]{target}" for
                         i, sentence
                         in enumerate(sentences)]


        result = self.get_input_token(sentences, max_seq_len)

        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        # print('--------')
        # print('here is the joint！')
        # print(sentences[0])
        # print(knowledges[0])
        # print(result)
        # print('--------')

        return seqs, seq_masks, seq_segments

    # max_seq_len总长，max_seq_num每句句长 知识处理
    def get_input2(self, sentences, max_seq_len=32, max_seq_num=6, target=None):
        sentences_sep = []
        for each in sentences:
            len_each = len(each)
            if len_each<=1:
                continue
            elif len_each < max_seq_num:
                each = each + [''] * (max_seq_num - len_each)
            else:
                each = each[:max_seq_num]
            sentences_sep.append(each)

        knowledges = []
        # # print('这里是sentences_sep')
        # # print(sentences_sep)
        # # print(type(sentences_sep))
        for sentences in sentences_sep:
            each_knowledge = []
            result = self.get_input_token(sentences, max_seq_len)
            for each in result:
                each_knowledge.append(list(each))
            knowledges.append(each_knowledge)
        return knowledges

    # 原版句子get_input3
    def get_input3(self, sentences, max_seq_len=128, knowledges=None, num=None, target=None):
        for i in range(len(sentences)):
            if type(sentences[i]) == float:
                sentences[i] = ''
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments



    def trunate_and_pad(self, seq, max_seq_len):
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CoAttention(nn.Module):
    def __init__(self, device, latent_dim=200):
        super(CoAttention, self).__init__()

        self.linearq = nn.Linear(latent_dim, latent_dim)
        self.lineark = nn.Linear(latent_dim, latent_dim)
        self.linearv = nn.Linear(latent_dim, latent_dim)

    def forward(self, sentence_rep, comment_rep, labels):
        query = self.linearq(sentence_rep)
        key = self.lineark(comment_rep)
        value = self.linearv(comment_rep)

        alpha_mat = torch.matmul(query, key.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, value).squeeze(1)

        return x

# 旧版
class MultiHeadAttention1(nn.Module):

    def __init__(self, device, latent_dim=64, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(latent_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(latent_dim, self.all_head_size)
        self.key = nn.Linear(latent_dim, self.all_head_size)
        self.value = nn.Linear(latent_dim, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, sentence_rep, comment_rep, labels):
        query_layer = self.transpose_for_scores(self.query(sentence_rep))
        key_layer = self.transpose_for_scores(self.key(comment_rep))
        value_layer = self.transpose_for_scores(self.value(comment_rep))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


# 新版
class MultiHeadAttention(nn.Module):

    def __init__(self, device, latent_dim=64, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(latent_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(latent_dim, self.all_head_size)
        self.key = nn.Linear(latent_dim, self.all_head_size)
        self.value = nn.Linear(latent_dim, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, sentence_rep, comment_rep, labels):
        query_layer = self.transpose_for_scores(self.query(comment_rep))  # swap here
        key_layer = self.transpose_for_scores(self.key(sentence_rep))  # swap here
        value_layer = self.transpose_for_scores(self.value(sentence_rep))  # swap here

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply attention to the value layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # Aggregate across the comment_rep dimension to ensure the output has the same batch size as sentence_rep
        context_layer = context_layer.mean(dim=2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


def simcse_sup_loss(y_pred,label,temp=0.05):
    """
    有监督的损失函数
    """
    # print('==========================')
    # print(label)
    # print('--------------------------')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_true = []
    batch_len = len(label)
    for i in range(batch_len):
        one_sim_lst = []
        for j in range(batch_len):
            if label[i].equal(label[j]):
                one_sim_lst.append(1)
            else:
                one_sim_lst.append(0)
        one_sim_lst[i]=0
        y_true.append(one_sim_lst)
        # print(y_true)
    y_true = torch.tensor(y_true,device=DEVICE).type(torch.float)


    # batch 内两两计算相似度, 得到相似度矩阵(对角矩阵)

    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12

    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss




# TexATT
class BertForSequenceClassification9(nn.Module):
    def __init__(self, model_path, num_labels=2, num_each_class=None, device='cuda'): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_path)
        # self.latent_dim = 768
        # self.latent_dim = 1024
        self.latent_dim = self.bert.config.hidden_size
        try:
            # AB 0.3
            # LA 0.5 HC
            self.dropout = nn.Dropout(0.3)

            # self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob).3
        except:
            self.dropout = nn.Dropout(self.bert.config.dropout)



        # simple att
        # self.coattention = CoAttention(device, self.latent_dim)
        # multi head att
        self.coattention = MultiHeadAttention(device, self.latent_dim,num_heads=2)


        self.classifier = nn.Linear(self.latent_dim * 2, num_labels)

        self.device = device
        self.num_each_class = num_each_class
        # nn.init.xavier_normal_(self.classifier.weight)
        # self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2)
        # self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=2)
        self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=2)
        self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = self.bert.config

        # Define the adapter module
        # self.adapter = nn.Sequential(
        #     nn.Linear(self.latent_dim, 64),  # adapter size 64
        #     nn.ReLU(),
        #     nn.Linear(64, self.latent_dim),
        # )

        self.config.pooling = 'cls'

    def forward_once(self, input_ids, attention_mask=None,token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)



        if self.config.pooling == 'cls':
            pooled_output1= outputs.last_hidden_state[:, 0]  # [batch, 768]

        elif self.config.pooling == 'pooler':
            pooled_output1= outputs.pooler_output  # [batch, 768]

        elif self.config.pooling == 'last-avg':
            last = outputs.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pooled_output1= torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        elif self.config.pooling == 'first-last-avg':
            first = outputs.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = outputs.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            pooled_output1= torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]





        # print(pooled_output)
        # 这里的drop改掉了
        pooled_output = self.dropout(outputs[1])

        # Apply the adapter module to the output of Bert
        # pooled_output = self.adapter(pooled_output)

        return pooled_output,pooled_output1


        # return self.bert(input_ids, token_type_ids, attention_mask)[1]



    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None, kg_labels=None):


        # print(labels)
        # print('labels')

        # forward pass of input 1
        output1, simcse_output1 = self.forward_once(batch_seqs, batch_seq_masks, batch_seq_segments)

        # output1 = self.dropout(output1)
        output1 = torch.unsqueeze(output1, 1)



        # forward pass of input 2
        knowledges = batch_knowledges.cpu()
        knowledges = knowledges.numpy().tolist()
        tmp = []
        # each是文本（编码）
        for each in knowledges:
            batch_seqs_k = [i[0] for i in each]
            batch_seq_masks_k = [i[1] for i in each]
            batch_seq_segments_k = [i[2] for i in each]
            # 换上装备上
            t_batch_seqs_k = torch.tensor(batch_seqs_k, dtype=torch.long).to(self.device)
            t_batch_seq_masks_k = torch.tensor(batch_seq_masks_k, dtype=torch.long).to(self.device)
            t_batch_seq_segments_k = torch.tensor(batch_seq_segments_k, dtype=torch.long).to(self.device)
            # 再次计算损失，基于知识
            output2,_ = self.forward_once(t_batch_seqs_k, t_batch_seq_masks_k, t_batch_seq_segments_k)
            tmp.append(output2)

        k_emb = torch.stack(tmp, dim=0)
        k_emb = self.tf_encoder(k_emb)
        # k_emb = self.dropout(k_emb)

        #aaaf
        pooled_output = self.coattention(output1, k_emb, labels)
        # saaf
        # pooled_output = self.coattention(k_emb, output1, labels)
        # print('-----shape here--------')
        # print(pooled_output.shape)

        # Apply the adapter module to the output of coattention
        # pooled_output = self.adapter(pooled_output)
        # print(pooled_output.squeeze().shape, output1.squeeze(1).shape)

        # pooled_output is reshaped to have dimensions [batch_size, feature_size]
        pooled_output = pooled_output.view(-1, pooled_output.size(-1))

        # Now we can concatenate along dimension 1
        # print('-----shape here!--------')
        # print(output1.shape)
        # print(pooled_output.shape)
        pooled_output = torch.cat([output1.squeeze(1), pooled_output], dim=1)

        # print(pooled_output.shape)

        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels>2:
                # loss_func_name = 'FL'

                # loss_func_name = 'CBloss'
                # loss_func_name = 'R-BCE-Focal'
                # loss_func_name = 'CBloss-ntr'
                # loss_func_name = 'DBloss'
                loss_func_name = 'BCE'

                if loss_func_name == 'BCE':
                    loss_fct = CrossEntropyLoss()
                    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)
                    # print('print(logits.double())')
                    # print(logits.double())
                    # print(logits.double().shape)
                    # print(labels_one_hot.double())
                    # print(labels_one_hot.double().shape)

                    loss = loss_fct(logits.double(), labels_one_hot.double())

                    # loss = loss + 0.1*simcse_sup_loss(simcse_output1, labels_one_hot)


                elif self.num_each_class:
                    train_num = sum(self.num_each_class)
                    # print(self.num_each_class)
                    loss_fct = loss_choice(loss_func_name, self.num_each_class, train_num, model_config=None)
                    # print(logits)
                    # print(logits.shape)
                    # print(labels.shape)
                    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)
                    loss = loss_fct(logits.double(), labels_one_hot.double())

                    # loss = loss + 0.1*simcse_sup_loss(simcse_output1, labels_one_hot)

                else:
                    loss_fct = FocalLoss(self.num_labels)
                # loss_fct = FocalLoss(self.num_labels)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = FocalLoss(self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # print(loss)
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True



class LLM:
    def __init__(self, model_path, device="cuda"):
        if 'yuanzh' in model_path:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model_type = 'chatyuan'
        elif 'moss' in model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            self.model_type = 'moss'
        elif 'llama' in model_path or 'vicuna' in model_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self.model = LlamaForCausalLM.from_pretrained(model_path)
            self.model_type = 'llama'

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            self.model_type = 'glm'
        if type(device)==str:
            if device=="cuda":
                self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.model.to(self.device)

    def preprocess(self, text):
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        return text

    def postprocess(self, text):
        return text.replace("\\n", "\n").replace("\\t", "\t")

    def answer(self, text, sample=True, top_p=1, temperature=0.7):
        if self.model_type=='chatyuan':
            '''sample：是否抽样。生成任务，可以设置为True;
            top_p：0-1之间，生成的内容越多样'''
            text = self.preprocess(text)
            encoding = self.tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(
                self.device)
            if not sample:
                out = self.model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                                     num_beams=1, length_penalty=0.6)
            else:
                out = self.model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                                     do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
            out_text = self.tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
            return self.postprocess(out_text[0])

        elif self.model_type=='moss':
            text = self.preprocess(text)
            inputs = self.tokenizer(text, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].cuda()
            outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.8, repetition_penalty=1.02,
                                     max_new_tokens=256)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self.postprocess(response)
        elif self.model_type=='llama':
            input_text = self.preprocess(text)
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(
                self.device)
            output_ids = self.model.generate(input_ids, max_new_tokens=256, temperature=temperature, top_p=0.8)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return self.postprocess(output_text)

        else:
            text = self.preprocess(text)
            response, _ = self.model.chat(self.tokenizer, text, history=[],temperature=temperature)
            return self.postprocess(response)



    def extract_label(self, text):
        pattern = r"(FAVOR|NONE|AGAINST|积极|消极|中立|反对|认可)"
        match = re.search(pattern, text)
        if match:
            label_dict = {
                "AGAINST": 0, "反对": 0, "消极": 0, "嘲讽":0, "嘲笑":0, "反讽":0, "against":0,
                "NONE": 1, "中立": 1,
                "FAVOR": 2, "认可": 2, "积极": 2, "满意": 2, "favor":2
            }
            return label_dict[match.group(0)]
        else:
            # 如果没有匹配到任何标签词语，返回一个默认的标签
            return 1  # 或者你可以选择抛出一个错误，例如：raise ValueError("No label found in text")






def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path
    parser.add_argument("--output_model_path", default="./models/pytorch_model.bin", type=str,
                        help="Path of the output models.")
    parser.add_argument("--output_lossfig_path", default="./models/loss.png", type=str,
                        help="Path of the output models.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--kg_seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--kg_each_seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--cat_num", type=int, default=1,
                        help="cat kg num for texts.")

    # generate data
    parser.add_argument("--n_for_combining", type=int, default=2,
                        help="Generate the sample with n data.")

    parser.add_argument("--g_m", type=int, default=300,
                        help="Generate m samples for a topic and a stance.")


    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=15,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")

    parser.add_argument(
        "--do_train",
        type=bool,
        default=False,
        help="do train or not",
    )

    parser.add_argument(
        "--llm",
        type=bool,
        default=False,
        help="use llm or not",
    )

    parser.add_argument(
        "--super_xgboost",
        type=bool,
        default=False,
        help="use super_xgboost or not",
    )

    parser.add_argument(
        "--only_knn",
        type=bool,
        default=False,
        help="only use knn or not",
    )

    parser.add_argument(
        "--data_path", type=str, default=None, help="the path of profile with dev.csv, test.csv, train.csv."
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="the path of profile with models"
    )
    parser.add_argument(
        "--model_path_2", type=str, default=None, help="the path of profile with models"
    )

    parser.add_argument("--k", type=int, default=3, help="k of KNN")


    parser.add_argument("--c", type=int, default=2, help="the number of the class")

    args = parser.parse_args()

    def set_seed(seed=666):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    # 3 class
    # 三分类
    def count_label_ratio3(df, column='label'):
        total = len(df)
        count_0 = len(df[df[column] == 0])
        count_1 = len(df[df[column] == 1])
        count_2 = len(df[df[column] == 2])

        return total,[count_0,count_1,count_2]

    # four分类
    def count_label_ratio2(df, column='label'):
        total = len(df)
        count_0 = len(df[df[column] == 0])
        count_1 = len(df[df[column] == 1])
        count_2 = len(df[df[column] == 2])
        count_3 = len(df[df[column] == 3])

        return total,[count_0,count_1,count_2,count_3]


    # 二分类
    def count_label_ratio4(df, column='label'):
        total = len(df)
        count_0 = len(df[df[column] == 0])
        count_1 = len(df[df[column] == 1])

        return total, [count_0, count_1]

    # def count_label_ratio(df, n_classes=3, column='label'):
    #     # n_classes=4
    #     total = len(df)
    #     counts = [len(df[df[column] == i]) for i in range(n_classes)]
    #
    #     return total, counts

    def count_label_ratio(df, n_classes=5, column='label', dummy_sample_count=10):
        """
        Counts the number of instances per class and sets a specific sample count for classes with 0 instances.

        Args:
        - df: Pandas DataFrame containing the dataset.
        - n_classes: The total number of classes, including original and dummy classes.
        - column: The name of the column in df that contains class labels.
        - dummy_sample_count: The sample count to set for classes that have 0 instances.

        Returns:
        - total: The total number of actual instances, not including adjustments for classes with 0 instances.
        - counts: A list containing the number of instances for each class, with adjustments for classes with 0 instances.
        """
        # Count instances for each of the classes
        counts = [len(df[df[column] == i]) for i in range(n_classes)]

        # Set a specific sample count for classes with 0 instances
        adjusted_counts = [count if count > 0 else dummy_sample_count for count in counts]

        # Calculate the total number of actual instances (not including dummy sample counts)
        total = sum(count if count > 0 else 0 for count in counts)

        return total, adjusted_counts

    def label_map2num(labels):
        if isinstance(labels[0], str):
            unique_labels = set(labels)
            if len(unique_labels) == 2:
                label_map = {"AGAINST": 0, "FAVOR": 1}
            elif len(unique_labels) == 3:
                label_map = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}
            else:
                raise ValueError("Unexpected number of unique labels")

            # 根据映射字典更新标签
            labels = [label_map[label] for label in labels]
        return labels

    set_seed(args.seed)

    if args.data_path == None:
        args.data_path = '/home/yindechun/Desktop/yanyu/data/zhihu/ATTTEX'
        # /home/yindechun/Desktop/yanyu
    # 读取数据
    # train = pd.read_csv('data/train.tsv', encoding='utf-8', sep='\t')
    # dev = pd.read_csv('data/dev.tsv', encoding='utf-8', sep='\t')
    # test = pd.read_csv('data/test.tsv', encoding='utf-8', sep='\t')


    rawdata = open(os.path.join(args.data_path,'train.csv'), 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    # for TP/PM
    # encoding = 'utf-8'

    # for PM
    # train = pd.read_excel(os.path.join(args.data_path, 'train.xlsx'))
    train = pd.read_csv(os.path.join(args.data_path,'train.csv'), encoding=encoding)


    dev = pd.read_csv(os.path.join(args.data_path,'dev.csv'), encoding=encoding)
    test = pd.read_csv(os.path.join(args.data_path,'test.csv'), encoding=encoding)

    class_num = train['label'].nunique()  # 获取唯一类别的数量

    # dummy clas
    class_num+=4
    # kg 原版
    # kg = pd.read_excel(os.path.join(args.data_path,'related_topic.xlsx'))
    # kg = kg['aspect'].tolist()
    # kg = dict(zip(kg, kg))

    kg = pd.read_excel(os.path.join(args.data_path, 'related_topic.xlsx'))


    train_total, num_each_class = count_label_ratio(train,n_classes=class_num)
    # Load bert vocabulary and tokenizer
    # bert_config = BertConfig('bert_model/bert_config.json')
    # BERT_MODEL_PATH = r'E:\models\white_model\chinesebert'
    BERT_MODEL_PATH = args.model_path
    BERT_MODEL_PATH_2 = args.model_path_2

    # kg = ['CnDbpedia','webhealth','Medical']
    # lookdown = False
    # graph = KnowledgeGraph(kg, lookdown)

    # knowledge_data_path = '/home/yindechun/Desktop/yanyu/data/sen_chinese_nlpcc_2016/kg_Iphone_SE.xlsx'
    # knowledge_df = pd.read_excel(knowledge_data_path)


    if not args.llm:
        graph = KnowledgeSentence(model_path=BERT_MODEL_PATH,kg=kg, k=args.k)
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        # 产生输入数据
        processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)
    else:
        graph = KnowledgeSentence(model_path=BERT_MODEL_PATH_2, kg=kg, k=args.k)
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH_2)
        # 产生输入数据
        processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)



    # super xgboost
    if args.super_xgboost:
        print("start to use super xgboost")

        train_label = train['label'].tolist()
        train_distances = graph.get_distances_idx(train['text_a'].tolist())
        stance_detector = graph.train_stance_detector(train_distances,train_label)
        test_distances = graph.get_distances_idx(test['text_a'].tolist())
        test_label = test['label'].tolist()

        graph.evaluate_stance(stance_detector,test_distances,test_label)


        return



    # train dataset
    # 1. get_input()返回的是seqs, seq_masks, seq_segments
    # Check if the file exists
    print("start to match train data")
    if os.path.exists('train_data.pkl'):
        # Load train_data
        with open('train_data.pkl', 'rb') as f:
            train_dataloder = pickle.load(f)
    else:

        if args.llm:
            # Get the knowledges
            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(train['text_a'].tolist())


            # Get the sentences
            sentences = train['text_a'].tolist()

            # Get the labels
            labels = train['label'].tolist()
            labels = label_map2num(labels)

            # Create a list of tuples, each tuple contains a sentence, its corresponding knowledges and label
            train_data = list(zip(sentences, knowledges_triples, labels))

            # 打印数据样例
            print('---------------------------')
            print('train_data[0]:', train_data[0])
            print('---------------------------')



            # Create a DataLoader
            train_dataloder = DataLoader(dataset=train_data, shuffle=True, batch_size=args.batch_size)

        else:

            # 2. get_input2()返回的是knowledges
            # 这个不是他妈的元组，是文本序列

            train_label = train['label'].tolist()
            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(train['text_a'].tolist(), label=train_label)

            # print(knowledges_triples)
            knowledges = processor.get_input2(knowledges_triples, max_seq_len=args.kg_seq_length, max_seq_num=args.kg_each_seq_length)

            seqs, seq_masks, seq_segments = processor.get_input(
                sentences=train['text_a'].tolist(), knowledges=knowledges_triples, max_seq_len=args.seq_length, num=args.cat_num, target=target)


            # 3. labels是train['label'].tolist()

            # print('---------------------------')
            # print('这里是知识序列')
            # print(knowledges_triples[0])
            # print(len(knowledges_triples[0]))
            # print('这里是知识')
            # print(knowledges[0])
            # print(len(knowledges))
            # print('这里是文段')
            # print(seqs[0])
            # print('---------------------------')

            labels = train['label'].tolist()
            labels = label_map2num(labels)


            t_seqs = torch.tensor(seqs, dtype=torch.long)
            t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
            t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
            t_knowledges = torch.tensor(knowledges, dtype = torch.long)
            t_topic_labels = torch.tensor(topic_labels, dtype = torch.float)
            t_labels = torch.tensor(labels, dtype = torch.long)


            # 4. t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels是torch.tensor类型
            # train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
            # 转换knowledge_labels为tensor
            t_knowledge_labels = torch.tensor(knowledge_labels, dtype=torch.long)
            # # print(t_knowledge_labels)
            # # 将所有tensor整合到TensorDataset中
            train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels, t_knowledge_labels, t_topic_labels)

            # print("t_seqs size: ", t_seqs.size())
            # print("t_seq_masks size: ", t_seq_masks.size())
            # print("t_seq_segments size: ", t_seq_segments.size())
            # print("t_knowledges size: ", t_knowledges.size())
            # print("t_labels size: ", t_labels.size())
            # print("t_knowledge_labels size: ", t_knowledge_labels.size())

            # 5. train_data是同样类型的TensorDatase
            train_sampler = RandomSampler(train_data)
            train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.batch_size)
            # Save train_data


        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_dataloder, f)


    print("start to match dev data")
    if os.path.exists('dev_data.pkl'):
        # Load train_data
        with open('dev_data.pkl', 'rb') as f:
            dev_dataloder = pickle.load(f)

    else:
        # dev dataset
        if args.llm:

            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(dev['text_a'].tolist())
            sentences = dev['text_a'].tolist()

            print('len here!!!! dev')
            print(len(sentences))
            print(len(knowledges_triples))

            labels = dev['label'].tolist()
            labels = label_map2num(labels)

            dev_data = list(zip(sentences, knowledges_triples, labels))
            dev_dataloder = DataLoader(dataset=dev_data, shuffle=True, batch_size=args.batch_size)



        else:

            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(dev['text_a'].tolist())

            knowledges = processor.get_input2(knowledges_triples,max_seq_len=args.kg_seq_length,max_seq_num=args.kg_each_seq_length)
            seqs, seq_masks, seq_segments = processor.get_input(
                sentences=dev['text_a'].tolist(), knowledges=knowledges_triples, max_seq_len=args.seq_length, target=target)
            labels = dev['label'].tolist()
            labels = label_map2num(labels)

            t_seqs = torch.tensor(seqs, dtype=torch.long)
            t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
            t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
            t_knowledges = torch.tensor(knowledges, dtype = torch.long)
            t_topic_labels = torch.tensor(topic_labels, dtype=torch.float)
            t_labels = torch.tensor(labels, dtype = torch.long)


            # dev_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
            # 转换knowledge_labels为tensor
            t_knowledge_labels = torch.tensor(knowledge_labels, dtype=torch.long)
            # # 将所有tensor整合到TensorDataset中
            dev_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels, t_knowledge_labels, t_topic_labels)


            dev_sampler = RandomSampler(dev_data)
            dev_dataloder = DataLoader(dataset=dev_data, sampler=dev_sampler, batch_size=args.batch_size)
        # Save dev_data
        with open('dev_data.pkl', 'wb') as f:
            pickle.dump(dev_dataloder, f)


    # test dataset
    print("start to match test data")
    if os.path.exists('test_data.pkl'):
        # Load train_data
        with open('test_data.pkl', 'rb') as f:
            test_dataloder = pickle.load(f)

    else:

        if args.llm:
            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(test['text_a'].tolist(), './kg/test_triples.txt', True,
                                                     test['label'].tolist())
            sentences = test['text_a'].tolist()
            labels = test['label'].tolist()
            labels = label_map2num(labels)
            test_data = list(zip(sentences, knowledges_triples, labels))
            test_dataloder = DataLoader(dataset=test_data, shuffle=True, batch_size=args.batch_size)


            # print('len here!!!! test')
            # print(len(sentences))
            # print(len(knowledges_triples))
            # print(knowledges_triples[0])

        else:
            knowledges_triples, knowledge_labels, target, topic_labels = graph.get_knowledge(test['text_a'].tolist(), './kg/test_triples.txt', True, test['label'].tolist())

            knowledges = processor.get_input2(knowledges_triples,max_seq_len=args.kg_seq_length,max_seq_num=args.kg_each_seq_length)
            seqs, seq_masks, seq_segments = processor.get_input(
                sentences=test['text_a'].tolist(), knowledges=knowledges_triples, max_seq_len=args.seq_length, target=target)
            labels = test['label'].tolist()
            labels = label_map2num(labels)
            t_seqs = torch.tensor(seqs, dtype=torch.long)
            t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
            t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
            t_knowledges = torch.tensor(knowledges, dtype = torch.long)
            t_topic_labels = torch.tensor(topic_labels, dtype=torch.float)
            t_labels = torch.tensor(labels, dtype = torch.long)

            # test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels)
            # 转换knowledge_labels为tensor
            t_knowledge_labels = torch.tensor(knowledge_labels, dtype=torch.long)
            # # 将所有tensor整合到TensorDataset中
            test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_knowledges, t_labels, t_knowledge_labels, t_topic_labels)


            test_sampler = RandomSampler(test_data)
            test_dataloder = DataLoader(dataset=test_data, sampler=test_sampler, batch_size=args.batch_size)

        # Save test_data
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(test_dataloder, f)


    # free graph
    del graph
    torch.cuda.empty_cache()



    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # build classification models
    # models = (bert_config, 2, device)
    if not args.llm:
        # print('debug num_each_class')
        # print(num_each_class)
        model = BertForSequenceClassification(BERT_MODEL_PATH, num_labels=len(num_each_class), num_each_class=num_each_class, device=device)

        if device == 'cuda':
            if torch.cuda.device_count() > 1:
                print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
        model = model.to(device)

    else:
        model = LLM(BERT_MODEL_PATH, device=device)

    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,  # 选择你的任务类型为序列分类
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1
    # )
    # model = get_peft_model(model, peft_config)




    # evaluation function
    def evaluate1(args, is_test, metrics='f1'):
        if is_test:
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        model.eval()

        # Get the number of classes from the model output or dataset
        num_classes = model.num_labels

        # Confusion matrix.
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        for i, batch_data in enumerate(dataset):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data
            with torch.no_grad():
                try:
                    logits, knn = model(
                        batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None,
                        kg_labels=batch_kg_labels)
                    pred_knn = knn.softmax(dim=1)
                    pred_ori = logits.softmax(dim=1)
                    pred = 0.5*pred_knn + 0.5*pred_ori
                    pred = pred.argmax(dim=1)
                except:

                    logits = model(
                        batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None,
                        kg_labels=batch_kg_labels)
                    pred = logits.softmax(dim=1).argmax(dim=1)
            gold = batch_labels
            labels = batch_labels.data.cpu().numpy()
            predic = pred.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")

        f1_avg = 0
        for i in range(confusion.size()[0]):
            if confusion[i, :].sum().item() == 0:
                p = 0
            else:
                p = confusion[i, i].item() / confusion[i, :].sum().item()
            if confusion[:, i].sum().item() == 0:
                r = 0
            else:
                r = confusion[i, i].item() / confusion[:, i].sum().item()
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
                f1_avg += f1
            print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))


        if num_classes>2:
            # Get classification report as a dict
            report_dict = sklearn.metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)

            # Add macro and micro average scores to the report dict
            report_dict['macro avg'] = {
                'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='macro'),
                'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='macro'),
                'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='macro'),
                'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
            }
            report_dict['micro avg'] = {
                'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='micro'),
                'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='micro'),
                'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='micro'),
                'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
            }

            # Convert the report dict to a DataFrame
            report_df = pd.DataFrame(report_dict).transpose()

            # Convert the DataFrame to a string
            report = report_df.to_string()
        else:
            report = sklearn.metrics.classification_report(labels_all, predict_all, digits=4)

        acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
        weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')

        if num_classes>2:
            macro_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='macro')
            mirco_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='micro')
        print('----------------------')
        print(report)
        print('----------------------')
        if metrics == 'Acc':
            return acc, None
        elif metrics == 'f1':
            if num_classes>2:
                # Get the f1-score of the first and third class
                f1_class1 = report_dict['0']['f1-score']
                f1_class3 = report_dict['2']['f1-score']
                # Calculate the average f1-score of the first and third class
                avg_f1 = (f1_class1 + f1_class3) / 2
                return avg_f1, None
                # return macro_f1, mirco_f1
            else:
                return weighted_f1, None
        else:
            return acc, None

    # no dummy
    def evaluate2(args, is_test, metrics='f1'):

        if is_test:
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)

        if not args.llm:


            model.eval()

            # Get the number of classes from the model output or dataset
            num_classes = model.num_labels

            # Confusion matrix.
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            correct = 0
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)

            for i, batch_data in enumerate(dataset):
                batch_data = tuple(t.to(device) for t in batch_data)
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels, batch_kg_labels, batch_tp_labels = batch_data
                # batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data
                with torch.no_grad():
                    try:
                        logits,knn = model(
                            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None, kg_labels=batch_kg_labels)
                        pred_knn = knn.softmax(dim=1)
                        pred_ori = logits.softmax(dim=1)
                        # no dummy eval 2
                        pred = 0.5*pred_knn + 0.5*pred_ori
                        pred = pred.argmax(dim=1)
                    except:
                        logits = model(
                            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None, kg_labels=batch_kg_labels)
                        pred = logits.softmax(dim=1).argmax(dim=1)

                gold = batch_labels
                labels = batch_labels.data.cpu().numpy()
                predic = pred.data.cpu().numpy()

                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)



                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                    if pred[j] != gold[j]:
                        with open('incorrect_predictions.txt', 'a') as f:
                            text = model.tokenizer.decode(batch_seqs[j], skip_special_tokens=True)
                            logits_str = ', '.join([f"{x:.4f}" for x in logits[j].cpu().numpy()])
                            f.write(f"Text: {text}\nPredicted: {pred[j]}\nActual: {gold[j]}\nLogits: {logits_str}\n\n")

                correct += torch.sum(pred == gold).item()

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            f1_avg = 0
            for i in range(confusion.size()[0]):
                if confusion[i, :].sum().item() == 0:
                    p = 0
                else:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                if confusion[:, i].sum().item() == 0:
                    r = 0
                else:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                if (p + r) == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                    f1_avg += f1
                print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))


            if num_classes>2:
                # Get classification report as a dict
                report_dict = sklearn.metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)

                # Add macro and micro average scores to the report dict
                report_dict['macro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='macro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='macro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='macro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }
                report_dict['micro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='micro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='micro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='micro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }

                # Convert the report dict to a DataFrame
                report_df = pd.DataFrame(report_dict).transpose()

                # Convert the DataFrame to a string
                report = report_df.to_string()
            else:
                report = sklearn.metrics.classification_report(labels_all, predict_all, digits=4)

            acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
            weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')
            if num_classes>2:
                macro_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='macro')
                mirco_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='micro')
            print('----------------------')
            print(report)
            print('----------------------')
            if metrics == 'Acc':
                return acc, None
            elif metrics == 'f1':
                if num_classes>2:
                    # Get the f1-score of the first and third class
                    f1_class1 = report_dict['0']['f1-score']
                    f1_class3 = report_dict['2']['f1-score']
                    # Calculate the average f1-score of the first and third class
                    avg_f1 = (f1_class1 + f1_class3) / 2
                    return avg_f1,None
                    # return mirco_f1, macro_f1
                else:
                    return weighted_f1, None
            else:
                return acc, None

        # -------------------------------------
        # large language model~~~~
        else:
            correct = 0

            # Define the class labels
            class_labels = [0, 1, 2]

            # Get the number of classes from the class labels
            num_classes = len(class_labels)

            # Confusion matrix.
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

            predict_all = []
            labels_all = []
            all_ = 0
            for i, batch_data in enumerate(dataset):
                batch_data = tuple(t for t in batch_data)
                batch_sentences, batch_knowledges, batch_labels, batch_kg_labels = batch_data
                # batch_sentences, batch_knowledges, batch_labels = batch_data

                # 预测
                # Get predictions for each sentence in the batch
                pred = []
                batch_knowledges = list(map(list, zip(*batch_knowledges)))
                # print('batch_knowledges')
                # print(len(batch_knowledges))
                # print(len(batch_knowledges[0]))

                for sentence, knowledges in zip(batch_sentences, batch_knowledges):
                    # chatyuan
                    # input_text = "用户：分类以下文本态度：”"  + "\n“" + sentence.replace(' ','') + "” \n小元：在" + ",".join(knowledges[:1]) + ",文本的态度（积极、消极、中立）是:"
                    # input_text = "用户：对于文本：“"+ sentence.replace(' ','') + "” \n小元：总的来说，文本对于Iphone SE的态度（积极、消极、中立）是"
                    # input_text = "用户：对于文本：“"+ sentence.replace(' ','') + "” \n小元：总的来说，文本对于Iphone SE的态度（积极、消极、中立、反讽）是"
                    #chatglm
                    input_text = "用户：分类以下文本态度：“"+ sentence.replace(' ','') + "” \n总的来说，文本对于Iphone SE的态度（积极、消极、中立）是"

                    #moss

                    output_text = model.answer(input_text, temperature=0.1)
                    all_+=1
                    pred_ = model.extract_label(output_text)


                    print(f'start ans:------------------')
                    print('text:----')
                    print(sentence)
                    print('output_text:----')
                    print(output_text)
                    print('batch_knowledges:----')
                    print(knowledges)
                    print('>>>>>pred_res:',pred_)
                    print('end ans------------------')
                    pred.append(pred_)




                gold = [class_labels[int(i.item())] for i in batch_labels]
                print('start--------gl------------')
                print(all_)
                print(gold)
                print(pred)
                print('end--------gl------------')

                labels_all.extend(gold)
                predict_all.extend(pred)
                assert len(labels_all) == len(
                    predict_all), f"Lengths of labels_all and predict_all are not the same: {len(labels_all)} vs {len(predict_all)}, and the length of batch_sentences is {len(batch_knowledges)}, batch_knowledges's is {len(batch_knowledges)}."


                for j in range(len(pred)):
                    confusion[class_labels.index(pred[j]), class_labels.index(gold[j])] += 1
                    if pred[j] != gold[j]:
                        with open('incorrect_predictions.txt', 'a',encoding='utf-8') as f:
                            text = batch_sentences[j]
                            f.write(
                                f"Text: {text}\nPredicted: {pred[j]}\nActual: {gold[j]}\nLogits:\n\n")

                correct += sum([1 if pred[j] == gold[j] else 0 for j in range(len(pred))])

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            f1_avg = 0
            for i in range(confusion.size()[0]):
                if confusion[i, :].sum().item() == 0:
                    p = 0
                else:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                if confusion[:, i].sum().item() == 0:
                    r = 0
                else:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                if (p + r) == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                    f1_avg += f1
                print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))


            if num_classes>2:
                # Get classification report as a dict
                report_dict = sklearn.metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)

                # Add macro and micro average scores to the report dict
                report_dict['macro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='macro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='macro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='macro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }
                report_dict['micro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='micro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='micro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='micro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }

                # Convert the report dict to a DataFrame
                report_df = pd.DataFrame(report_dict).transpose()

                # Convert the DataFrame to a string
                report = report_df.to_string()
            else:
                report = sklearn.metrics.classification_report(labels_all, predict_all, digits=4)

            acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
            weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')
            if num_classes>2:
                macro_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='macro')
                mirco_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='micro')
            print('----------------------')
            print(report)
            print('----------------------')
            if metrics == 'Acc':
                return acc, None
            elif metrics == 'f1':
                if num_classes>2:
                    # Get the f1-score of the first and third class
                    f1_class1 = report_dict['0']['f1-score']
                    f1_class3 = report_dict['2']['f1-score']
                    # Calculate the average f1-score of the first and third class
                    avg_f1 = (f1_class1 + f1_class3) / 2
                    return avg_f1,None
                    # return mirco_f1, macro_f1
                else:
                    return weighted_f1, None
            else:
                return acc, None

    def evaluate(args, is_test, metrics='f1'):

        if is_test:
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)

        if not args.llm:

            model.eval()

            # Get the number of classes from the model output or dataset
            num_classes = model.num_labels

            # Confusion matrix.

            correct = 0
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)

            for i, batch_data in enumerate(dataset):
                batch_data = tuple(t.to(device) for t in batch_data)
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels, batch_kg_labels, batch_tp_labels = batch_data
                # batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data
                with torch.no_grad():
                    try:
                        logits, knn = model(
                            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None,
                            kg_labels=batch_kg_labels)
                        pred_knn = knn.softmax(dim=1)
                        pred_ori = logits.softmax(dim=1)

                        # 创建一个与pred_knn相同形状的全0 tensor
                        expanded_tp_labels = torch.zeros_like(pred_knn)
                        # 将batch_tp_labels的内容复制到expanded_tp_labels的前三个类别中
                        expanded_tp_labels[:, :batch_tp_labels.size(1)] = batch_tp_labels

                        # print(expanded_tp_labels)

                        # true eval dummy
                        #
                        # pred = 0.5 * pred_knn + 0.5 * pred_ori
                        pred = 0.3 * pred_knn + 0.2 * expanded_tp_labels + 0.5 * pred_ori
                        pred = pred.argmax(dim=1)
                    except:
                        logits = model(
                            batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, labels=None,
                            kg_labels=batch_kg_labels)
                        pred = logits.softmax(dim=1).argmax(dim=1)
                gold = batch_labels


                labels_all = np.append(labels_all, batch_labels.data.cpu().numpy())
                predict_all = np.append(predict_all, pred.data.cpu().numpy())
                correct += torch.sum(pred == gold).item()


            # After collecting all labels and predictions
            # Find unique classes present in labels_all
            unique_classes = sorted(set(labels_all))
            num_classes = len(unique_classes)

            # Initialize confusion matrix with actual number of classes
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

            # for j in range(pred.size()[0]):
            #     confusion[pred[j], gold[j]] += 1
            #     if pred[j] != gold[j]:
            #         with open('incorrect_predictions.txt', 'a') as f:
            #             text = model.tokenizer.decode(batch_seqs[j], skip_special_tokens=True)
            #             logits_str = ', '.join([f"{x:.4f}" for x in logits[j].cpu().numpy()])
            #             f.write(f"Text: {text}\nPredicted: {pred[j]}\nActual: {gold[j]}\nLogits: {logits_str}\n\n")

            # Loop through each prediction in the batch
            for index in range(len(predict_all)):
                pred_label = predict_all[index]
                true_label = labels_all[index]
                if pred_label < num_classes and true_label < num_classes:
                    confusion[pred_label, true_label] += 1

            # Handle incorrect predictions
            for j in range(len(predict_all)):
                if predict_all[j] != labels_all[j]:
                    # Open and append to the incorrect predictions file
                    with open('incorrect_predictions.txt', 'a', encoding='utf-8') as f:
                        text = model.tokenizer.decode(batch_seqs[j % len(batch_seqs)],
                                                      skip_special_tokens=True)  # Ensure index loops within single batch size
                        logits_str = ', '.join([f"{x:.4f}" for x in logits[
                            j % len(logits)].cpu().numpy()])  # Ensure index loops within single batch size
                        f.write(
                            f"Text: {text}\nPredicted: {predict_all[j]}\nActual: {labels_all[j]}\nLogits: {logits_str}\n\n")


            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            f1_avg = 0
            for i in range(confusion.size()[0]):
                if confusion[i, :].sum().item() == 0:
                    p = 0
                else:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                if confusion[:, i].sum().item() == 0:
                    r = 0
                else:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                if (p + r) == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                    f1_avg += f1
                print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))

            if num_classes > 2:
                report_dict = sklearn.metrics.classification_report(
                    labels_all, predict_all, labels=unique_classes, digits=4, output_dict=True
                )

                # Calculate macro and micro averages for the report
                report_dict['macro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, labels=unique_classes,
                                                                 average='macro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, labels=unique_classes,
                                                           average='macro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, labels=unique_classes,
                                                         average='macro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in unique_classes])
                }
                report_dict['micro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, labels=unique_classes,
                                                                 average='micro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, labels=unique_classes,
                                                           average='micro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, labels=unique_classes,
                                                         average='micro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in unique_classes])
                }

                # Convert the report dict to a DataFrame and then to a string
                report_df = pd.DataFrame(report_dict).transpose()
                report = report_df.to_string()
            else:
                report = sklearn.metrics.classification_report(labels_all, predict_all, digits=4)

            # Print out the complete report and other metrics
            print('----------------------')
            print(report)
            print('----------------------')

            # Return the required metric
            if metrics == 'Acc':
                return sklearn.metrics.accuracy_score(labels_all, predict_all), None
            elif metrics == 'f1':
                if num_classes > 2:
                    # Adjusted to calculate f1 scores only for existing classes
                    f1_scores = [report_dict[str(label)]['f1-score'] for label in unique_classes]
                    avg_f1 = np.mean(f1_scores)
                    return avg_f1, None
                else:
                    return sklearn.metrics.f1_score(labels_all, predict_all, average='weighted'), None
            else:
                return sklearn.metrics.accuracy_score(labels_all, predict_all), None

        # -------------------------------------
        # large language model~~~~
        else:
            correct = 0

            # Define the class labels
            class_labels = [0, 1, 2]

            # Get the number of classes from the class labels
            num_classes = len(class_labels)

            # Confusion matrix.
            confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

            predict_all = []
            labels_all = []
            all_ = 0
            for i, batch_data in enumerate(dataset):
                batch_data = tuple(t for t in batch_data)
                batch_sentences, batch_knowledges, batch_labels, batch_kg_labels = batch_data
                # batch_sentences, batch_knowledges, batch_labels = batch_data

                # 预测
                # Get predictions for each sentence in the batch
                pred = []
                batch_knowledges = list(map(list, zip(*batch_knowledges)))
                # print('batch_knowledges')
                # print(len(batch_knowledges))
                # print(len(batch_knowledges[0]))

                for sentence, knowledges in zip(batch_sentences, batch_knowledges):
                    # chatyuan
                    # input_text = "用户：分类以下文本态度：”"  + "\n“" + sentence.replace(' ','') + "” \n小元：在" + ",".join(knowledges[:1]) + ",文本的态度（积极、消极、中立）是:"
                    # input_text = "用户：对于文本：“"+ sentence.replace(' ','') + "” \n小元：总的来说，文本对于Iphone SE的态度（积极、消极、中立）是"
                    # input_text = "用户：对于文本：“"+ sentence.replace(' ','') + "” \n小元：总的来说，文本对于Iphone SE的态度（积极、消极、中立、反讽）是"
                    # chatglm
                    input_text = "用户：分类以下文本态度：“" + sentence.replace(' ',
                                                                              '') + "” \n总的来说，文本对于Iphone SE的态度（积极、消极、中立）是"

                    # moss

                    output_text = model.answer(input_text, temperature=0.1)
                    all_ += 1
                    pred_ = model.extract_label(output_text)

                    print(f'start ans:------------------')
                    print('text:----')
                    print(sentence)
                    print('output_text:----')
                    print(output_text)
                    print('batch_knowledges:----')
                    print(knowledges)
                    print('>>>>>pred_res:', pred_)
                    print('end ans------------------')
                    pred.append(pred_)

                gold = [class_labels[int(i.item())] for i in batch_labels]
                print('start--------gl------------')
                print(all_)
                print(gold)
                print(pred)
                print('end--------gl------------')

                labels_all.extend(gold)
                predict_all.extend(pred)
                assert len(labels_all) == len(
                    predict_all), f"Lengths of labels_all and predict_all are not the same: {len(labels_all)} vs {len(predict_all)}, and the length of batch_sentences is {len(batch_knowledges)}, batch_knowledges's is {len(batch_knowledges)}."

                for j in range(len(pred)):
                    confusion[class_labels.index(pred[j]), class_labels.index(gold[j])] += 1
                    if pred[j] != gold[j]:
                        with open('incorrect_predictions.txt', 'a', encoding='utf-8') as f:
                            text = batch_sentences[j]
                            f.write(
                                f"Text: {text}\nPredicted: {pred[j]}\nActual: {gold[j]}\nLogits:\n\n")

                correct += sum([1 if pred[j] == gold[j] else 0 for j in range(len(pred))])

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            f1_avg = 0
            for i in range(confusion.size()[0]):
                if confusion[i, :].sum().item() == 0:
                    p = 0
                else:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                if confusion[:, i].sum().item() == 0:
                    r = 0
                else:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                if (p + r) == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                    f1_avg += f1
                print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / instances_num, correct, instances_num))

            if num_classes > 2:
                # Get classification report as a dict
                report_dict = sklearn.metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)

                # Add macro and micro average scores to the report dict
                report_dict['macro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='macro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='macro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='macro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }
                report_dict['micro avg'] = {
                    'precision': sklearn.metrics.precision_score(labels_all, predict_all, average='micro'),
                    'recall': sklearn.metrics.recall_score(labels_all, predict_all, average='micro'),
                    'f1-score': sklearn.metrics.f1_score(labels_all, predict_all, average='micro'),
                    'support': np.sum([report_dict[str(i)]['support'] for i in range(num_classes)])
                }

                # Convert the report dict to a DataFrame
                report_df = pd.DataFrame(report_dict).transpose()

                # Convert the DataFrame to a string
                report = report_df.to_string()
            else:
                report = sklearn.metrics.classification_report(labels_all, predict_all, digits=4)

            acc = sklearn.metrics.accuracy_score(labels_all, predict_all)
            weighted_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='weighted')
            if num_classes > 2:
                macro_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='macro')
                mirco_f1 = sklearn.metrics.f1_score(labels_all, predict_all, average='micro')
            print('----------------------')
            print(report)
            print('----------------------')
            if metrics == 'Acc':
                return acc, None
            elif metrics == 'f1':
                if num_classes > 2:
                    # Get the f1-score of the first and third class
                    f1_class1 = report_dict['0']['f1-score']
                    f1_class3 = report_dict['2']['f1-score']
                    # Calculate the average f1-score of the first and third class
                    avg_f1 = (f1_class1 + f1_class3) / 2
                    return avg_f1, None
                    # return mirco_f1, macro_f1
                else:
                    return weighted_f1, None
            else:
                return acc, None

    # training phase
    if args.do_train:
        if args.llm:

            # Define a new DataFrame to store the new samples
            new_samples_df = pd.DataFrame(columns=['order', 'text_a', 'label', 'Target'])

            # Check the column name for 'Target' and convert it to standard format
            target_column = None
            for column in train.columns:
                if column.lower() == 'target':
                    target_column = column
                    break

            if target_column is None:
                raise ValueError("No 'Target' column found in the data.")

            # Get unique label, target pairs
            unique_pairs = train[['label', target_column]].drop_duplicates().values

            # Define a dictionary to map the numerical labels to their textual representations
            # label_dict = {0: 'AGAINST', 1: 'NONE', 2: 'FAVOR'}
            label_dict = {0: '反对', 1: '中立', 2: '认可'}

            # Initialize a counter for the order
            counter = 0

            # For each unique label, target pair
            for label, target in tqdm(unique_pairs, desc='Generating new samples'):
                # Filter the DataFrame for rows with this label and target
                filtered_df = train[(train['label'] == label) & (train[target_column] == target)]
                filtered_df = filtered_df.reset_index()
                atti = label_dict[int(filtered_df.loc[0, 'label'])]

                # Randomly draw m samples
                for i in tqdm(range(args.g_m), desc='Randomly drawing samples'):
                    # Randomly select n rows from the filtered DataFrame
                    sample_df = filtered_df.sample(args.n_for_combining)

                    # Combine the selected samples into one sample
                    combined_text = '\n'.join(sample_df['text_a'])

                    # Generate the prompt
                    # 要求态度一致({label_dict[label]} 在话题"{target}"中):
                    # prompt = f'''请你融合以下文本中的信息，重新表述：{combined_text}'''
                    # prompt = f'''请你代表他们观点进行简短的发言：{combined_text}'''
                    # prompt = f'''生成相似言论：{combined_text}'''
                    prompt = f'''生成一句简短的{atti}态度言论：\n{combined_text}'''
                    # prompt = f'''Please generate a new statement btased on the following statements, requiring that the statement be consistent with the stance and attitude of the original statement.({label_dict[label]} in the target "{target}"):\n{combined_text}'''

                    # Generate a new sample
                    new_sample = model.answer(prompt)

                    # Print the prompt and the new sample
                    print(f'start prompt, No.{counter + 1}------------------------------------')
                    print('prompt')
                    print(prompt)
                    print('res')
                    print(new_sample)
                    # print(new_samples_df.head())
                    print('end prompt------------------------------------')

                    # Add the new sample to the new_samples_df DataFrame
                    new_samples_df.loc[counter] = [counter, new_sample, label, target]
                    counter += 1


            new_samples_df.to_csv('new_samples.csv', index=False)



        else:
            print("Start training.")
            instances_num = train.shape[0]
            batch_size = args.batch_size
            train_steps = int(instances_num * args.epochs_num / batch_size) + 1

            print("Batch size: ", batch_size)
            print("The number of training instances:", instances_num)


            # 待优化的参数
            # param_optimizer = list(models.named_parameters())
            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # optimizer_grouped_parameters = [
            #     {
            #         'params':
            #         [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            #         'weight_decay':
            #         0.01
            #     },
            #     {
            #         'params':
            #         [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            #         'weight_decay':
            #         0.0
            #     }
            # ]
            # optimizer = BertAdam(optimizer_grouped_parameters,
            #                     lr=args.learning_rate,
            #                     warmup=args.warmup,
            #                     t_total=train_steps)

            # Optimizer
            # Split weights in two groups, one with weight decay and the other not.


            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            no_decay = ["bias", "LayerNorm.weight"]
            # print(model.named_parameters())

            # model_params = list(model.named_parameters())
            # print("Parameter name \t\t requires_grad")
            # print("-" * 40)
            # for name, param in model_params:
            #     print(f"{name}\t\t{param.requires_grad}")


            optimizer_grouped_parameters = [
                {
                    'params':
                        [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay':
                        args.weight_decay
                },
                {
                    'params':
                        [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay':
                        0.0
                }
            ]



            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            # optimizer = Lion(optimizer_grouped_parameters, lr=args.learning_rate)
            # optimizer = Lion(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            # 存储每一个batch的loss
            all_loss = []
            all_acc = []
            total_loss = 0.0
            result = 0.0
            best_result = 0.0

            for epoch in range(1, args.epochs_num + 1):
                model.train()
                print('start epoch---------------------------------------')
                print(f'epoch:{epoch}')
                for step, batch_data in tqdm(enumerate(train_dataloder), total=len(train_dataloder)):
                    batch_data = tuple(t.to(device) for t in batch_data)
                    # batch_seps是编码后文本，batch_seq_masks是文本的mask，batch_seq_segments是文本的segment
                    # batch_labels是标签
                    # batch_knowledges是知识库，句子编码
                    batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels, batch_kg_labels, batch_tp_labels = batch_data
                    # batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels = batch_data
                    # 对标签进行onehot编码
                    # one_hot = torch.zeros(batch_labels.size(0), 2).long()
                    '''one_hot_batch_labels = one_hot.scatter_(
                        dim=1,
                        index=torch.unsqueeze(batch_labels, dim=1),
                        src=torch.ones(batch_labels.size(0), 2).long())
        
                    
                    logits = models(
                        batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
                    logits = logits.softmax(dim=1)
                    loss_function = CrossEntropyLoss()
                    loss = loss_function(logits, batch_labels)'''
                    # 输入了batch_knowledges

                    # 输入了batch_knowledges
                    # inputs = {
                    #     'input_ids': batch_seqs,
                    #     'attention_mask': batch_seq_masks,
                    #     'token_type_ids': batch_seq_segments,
                    #     'knowledge_ids': batch_knowledges,
                    #     'labels': batch_labels
                    # }


                    # loss = model(
                    #     batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels,None)
                    loss = model(
                        batch_seqs, batch_seq_masks, batch_seq_segments, batch_knowledges, batch_labels,batch_kg_labels)
                    # forward(self, input_ids, attention_mask=None, token_type_ids=None, batch_knowledges=None, labels=None, kg_labels=None):
                    # loss = model(**inputs)

                    loss.backward()
                    total_loss += loss.item()
                    if (step + 1) % 100 == 0:
                        print("Epoch id: {}, Training steps: {}, Avg loss: {:.4f}".format(epoch, step+1, total_loss / 100))
                        sys.stdout.flush()
                        total_loss = 0.
                    #print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, step+1, loss))
                    optimizer.step()
                    optimizer.zero_grad()


                print(f'end epoch{epoch}---------------------------------------')

                all_loss.append(total_loss)
                total_loss = 0.
                print("Start evaluation on dev dataset.")

                # 多分类这里来改
                result = evaluate(args, False, metrics='f1')[0]
                all_acc.append(result)
                if result > best_result:
                    best_result = result
                    #torch.save(models, open(args.output_model_path,"wb"))
                    #save_model(models, args.output_model_path)
                    torch.save(model.state_dict(), args.output_model_path)
                else:
                    # continue
                    pass

                print("Start evaluation on test dataset.")
                evaluate(args, True)

            print('all_loss:', all_loss)
            print('all_acc:', all_acc)



    # 测试阶段
    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    if not args.llm:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)



    '''
    print(loss_collect)
    plt.figure(figsize=(12,8))
    plt.plot(range(len(loss_collect)), loss_collect,'g.')
    plt.grid(True)
    plt.savefig(args.output_lossfig_path)'''

if __name__ == "__main__":
    main()