import math
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_class, d_model=512, dim_feedforward=2048, num_head=8, num_layers=6,
                 dropout=0.1, max_len=10000, activation: str = 'relu'):
        """
        Transformer
        :param vocab_size: 词表大小
        :param d_model: 词向量维度，相当于embedding_dim
        :param num_class: 分类类别数
        :param dim_feedforward: Encoder中前馈神经网络输出维度
        :param num_head: 多头注意力机制的头数
        :param num_layers: 编码器的层数
        :param dropout: 丢弃率
        :param max_len: 位置编码矩阵的最大长度
        :param activation: 激活函数
        """
        super(Transformer, self).__init__()
        self.embedding_dim = d_model
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        # 位置编码层
        self.position_embedding = PositionalEncoding(self.embedding_dim, dropout, max_len)
        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(d_model))
        # 分类层
        self.classifier = nn.Linear(d_model, num_class)

    def forward(self, inputs, lengths):
        # 数据处理与lstm相同，输出数据的第一个维度是批次，但这里需要将其转换为
        # Transformer Encoder所需要的第一个维度是序列长度，第二个维度是批次的形状
        inputs = torch.transpose(inputs, 0, 1)
        # 这里根据论文3.4部分的描述对原始词向量进行了缩放
        embeddings = self.embeddings(inputs.long()) * math.sqrt(self.embedding_dim)
        hidden_states = self.position_embedding(embeddings)
        # 根据批次中每个序列长度生成mask矩阵
        attention_mask = length_to_mask(lengths) == False
        memory = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        # 取最后一个时刻的输出作为分类层的输入
        memory = memory[-1, :, :]
        output = self.classifier(memory)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        """
        Transformer的位置编码层
        :param d_model: 输入词向量维度
        :param dropout: 丢弃率
        :param max_len: 位置编码矩阵的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]  对偶数进行位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 对奇数进行位置编码
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # 将pe注册为模型参数

    def forward(self, x):
        """
        前向传播过程
        :param x: [x_len, batch_size, embedding_dim]
        :return: [x_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]  # [x_len, batch_size, d_model] 输入的词向量与位置编码进行相加
        return self.dropout(x)


def length_to_mask(lengths):
    """
    将序列的长度转换成mask矩阵，忽略序列补齐后padding部分的信息
    :param lengths: [batch,]
    :return: batch * max_len
    """
    max_len = torch.max(lengths).long()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask



