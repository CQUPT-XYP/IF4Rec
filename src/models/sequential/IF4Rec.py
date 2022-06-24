from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

from models.BaseModel import SequentialModel
from utils import MyLayers as layers

'''
python main.py --model_name IF4Rec --lr 1e-4 --l2 1e-6
'''

class IF4Rec(SequentialModel):
    extra_log_args = ['emb_size', 'attn_size', 'K', 'encoder_layer', 'n_head', 'encoder_dropout', 'add_pos', 'add_multi']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        parser.add_argument('--encoder_layer', type=int, default=1,
                            help='Number of encoder layer.')
        parser.add_argument('--n_head', type=int, default=8,
                            help='Number of multi-attention head')
        parser.add_argument('--encoder_dropout', type=float, default=0.1,
                            help='encoder dropout')
        parser.add_argument('--add_multi', type=int, default=1,
                            help='Whether add multi-head.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size  # embedding size
        self.attn_size = args.attn_size  # attention size
        self.K = args.K  # num of interest
        self.add_pos = args.add_pos  # whether add position embedding
        self.max_his = args.history_max  # length of max_his
        self.encoder_layer = args.encoder_layer  # num of encoder layer
        self.n_head = args.n_head
        self.encoder_dropout = args.encoder_dropout
        self.add_multi = args.add_multi
        super().__init__(args, corpus)

        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num,
                                         self.emb_size)  # 杩欓噷鏄仛浜嗗鐞嗙殑锛屽疄闄呬笂鐨刬tem_num姣斿簭鍒楅暱搴﹀ぇ涓€锛屽洜姝ゅ彲浠ョ洿鎺ヨ祴鍊?

        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=self.n_head, dropout=self.encoder_dropout)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layer)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.n_head,
                                    dropout=self.encoder_dropout, kq_same=False)
            for _ in range(self.encoder_layer)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max] [256, 20]
        lengths = feed_dict['lengths']  # [batch_size] [256]
        batch_size, seq_len = history.shape  # batch_size:256; seq_len: 20

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[0, 1, 2, 3, 4], [0, 0, 0, 1, 2], [1, 2, 3, 4, 5]]
        if self.add_pos:
            position = (((lengths[:, None] + self.len_range[None, :seq_len]) - lengths[:, None]) * valid_his)
            pos_vectors = self.p_embeddings(position)
            his_pos_vectors = his_vectors + pos_vectors
        else:
            his_pos_vectors = his_vectors
        # his_pos_vectors : [256, 20, 64] the sequence user interacted sorted by time, 256 user, 20 item everyone, 64 is the dimension of item

        # Self-attention
        # first, linear transform, [256, 20, 64] -> [256, 20, 8] -> [256, 20, 2], activation is tan()
        # attn_score = self.W2(self.W1(his_pos_vectors).tanh())  # bsz, his_max, K [256, 20, 2]
        attn_score = self.W2(F.leaky_relu(self.W1(his_pos_vectors)))  # bsz, his_max, K [256, 20, 2]
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)  # let empty become -inf
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max [256, 2, 20]
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_pos_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(
            -2)  # bsz, K, emb; [256, 2, 64]

        # interest_vectors = self.encoder(interest_vectors)

        if self.add_multi:
            for block in self.transformer_block:
                interest_vectors = block(interest_vectors)

        # interest_vectors = self.extract_layer(interest_vectors)

        # dataframe = pd.DataFrame({'interest': interest_vectors.detach().numpy()},index=[0])
        # dataframe.to_csv("record.csv", index=False, sep=',')

        i_vectors = self.i_embeddings(i_ids)
        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0]  # bsz, emb
            target_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
            idx_select = target_pred.max(-1)[1]  # bsz
            user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)  # 鎶婃渶鍚庝竴涓淮搴﹀帇缂?
        else:
            prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
            prediction = prediction.max(-1)[0]  # bsz, -1

        return {'prediction': prediction.view(batch_size, -1)}
