# -*- encoding: utf-8 -*-
'''
@File    :   predict.py
@Time    :   2019/12/27 19:20:51
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   BERT-BILSTM-CRF FOR BER
'''

import torch
import os
import sys
# sys.path.append('/root/workspace/Bert-BiLSTM-CRF-pytorch')
from utils import tag2idx, idx2tag
from crf import Bert_BiLSTM_CRF
from pytorch_pretrained_bert import BertTokenizer
from typing import NamedTuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CRF_MODEL_PATH = '/root/workspace/Bert-BiLSTM-CRF-pytorch/checkpoints/finetuning/100.pt'
BERT_PATH = '/root/workspace/qa_project/chinese_L-12_H-768_A-12'

class CRF(object):
    def __init__(self, crf_model, bert_model, device='cpu'):
        self.device = torch.device(device)
        self.model = Bert_BiLSTM_CRF(tag2idx)
        self.model.load_state_dict(torch.load(crf_model))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        
    def predict(self, text):
        """Using CRF to predict label
        
        Arguments:
            text {str} -- [description]
        """
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        xx = self.tokenizer.convert_tokens_to_ids(tokens)
        xx = torch.tensor(xx).unsqueeze(0).to(self.device)
        _, y_hat = self.model(xx)
        pred_tags = []
        for tag in y_hat.squeeze():
            pred_tags.append(idx2tag[tag.item()])
        return pred_tags, tokens

    def parse(self, tokens, pred_tags):
        """Parse the predict tags to real word
        
        Arguments:
            x {List[str]} -- the origin text
            pred_tags {List[str]} -- predicted tags

        Return:
            entities {List[str]} -- a list of entities
        """
        entities = []
        entity = None
        for idx, st in enumerate(pred_tags):
            if entity is None:
                if st.startswith('B'):
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
            else:
                if st == 'O':
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities.append(name)
                    entity = None
                elif st.startswith('B'):
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities.append(name)
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
        return entities


crf = CRF(CRF_MODEL_PATH, BERT_PATH, 'cuda')

def get_crf_ners(text):
    # text = '罗红霉素和头孢能一起吃吗'
    pred_tags, tokens = crf.predict(text)
    entities = crf.parse(tokens, pred_tags)
    return entities


if __name__ == "__main__":
    text = '罗红霉素和头孢能一起吃吗'
    print(get_crf_ners(text))
