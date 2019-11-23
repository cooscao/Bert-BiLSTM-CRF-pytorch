import torch
import torch.nn as nn
from utils import tag2idx, idx2tag
from pytorch_pretrained_bert import BertModel
        

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('/root/workspace/qa_project/chinese_L-12_H-768_A-12')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)  #[128, 74, 768]
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning
        # self.bert.to(self.device)

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)  #[128, 74]
        y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]  # [128, 74, 768]

        if self.top_rnns:
            enc, _ = self.rnn(enc)  #[128, 74, 768]
        logits = self.fc(enc)  #[128, 74, 10]
        y_hat = logits.argmax(-1)  #[128, 74]
        return logits, y, y_hat

