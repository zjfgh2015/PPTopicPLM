import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    """
    TextCNN模型，用于文本分类任务。

    参数:
    - vocab_size: 词汇表大小
    - embedding_dim: 嵌入向量的维度
    - filter_size: 卷积核的大小
    - num_filter: 卷积核的数量
    - num_class: 输出的类别数
    """

    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(TextCNN, self).__init__()
        # 嵌入层，将词汇表大小映射到指定的嵌入维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 卷积层，Conv1d的输入是 (batch_size, embedding_dim, max_seq_len)
        # padding=1 是为了保持卷积操作后序列长度一致
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)

        # 激活函数 (ReLU)
        self.relu = nn.ReLU()

        # 全连接层，用于将卷积的输出映射到类别数上
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        # 输入的维度为 (batch_size, max_seq_len)
        embedding = self.embedding(inputs)
        # 经过嵌入层后，维度变为 (batch_size, max_seq_len, embedding_dim)

        # 交换维度，变为 (batch_size, embedding_dim, max_seq_len) 以适应 Conv1d 的输入要求
        embedding = embedding.permute(0, 2, 1)

        # 经过卷积和激活函数，输出维度为 (batch_size, num_filter, out_seq_len)
        convolution = self.relu(self.conv1d(embedding))

        # 经过最大池化，池化到每个卷积核的一个最大值，输出维度为 (batch_size, num_filter, 1)
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])

        # 降维，输出维度为 (batch_size, num_filter)，以适应全连接层的输入
        flatten = pooling.squeeze(dim=2)

        # 全连接层，将卷积结果映射到类别数，输出维度为 (batch_size, num_class)
        logits = self.linear(flatten)

        return logits
