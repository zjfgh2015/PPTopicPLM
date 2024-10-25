import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)  # 可选：在 LSTM 输出后加入 Dropout 防止过拟合
        self.output = nn.Linear(hidden_dim * 2, num_class)  # 双向 LSTM 输出维度为 hidden_dim * 2

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        # 使用 pack_padded_sequence 函数将变长序列打包
        x_pack = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(x_pack)
        # 拼接前向和反向的隐藏状态
        hidden = torch.cat([hn[0], hn[1]], dim=1)
        hidden = self.dropout(hidden)
        outputs = self.output(hidden)
        return outputs


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(BiGRU, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bi_gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)  # 可选：在 GRU 输出后加入 Dropout 防止过拟合
        self.output = nn.Linear(hidden_dim * 2, num_class)  # 双向 GRU 输出维度为 hidden_dim * 2

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        # 使用 pack_padded_sequence 函数将变长序列打包
        x_pack = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hn = self.bi_gru(x_pack)
        # 拼接前向和反向的隐藏状态
        hidden = torch.cat([hn[0], hn[1]], dim=1)
        hidden = self.dropout(hidden)
        outputs = self.output(hidden)
        return outputs
