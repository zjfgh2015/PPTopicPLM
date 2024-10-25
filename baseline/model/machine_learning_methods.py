import numpy as np
import pandas as pd
import torch
from torch import nn

def data_process_for_ml(data_filepath, bert_embedding_filepath, document_topics_filepath, manual_features_filepath):
    """
    加载并处理数据，合并BERT嵌入、文档主题分布和手工特征。

    参数:
    - data_filepath: 数据集路径（用于提取标签）
    - bert_embedding_filepath: BERT特征向量路径
    - document_topics_filepath: 文档主题分布路径
    - manual_features_filepath: 手工特征

    返回:
    - features: 拼接后的特征矩阵
    - labels: 数据集中的标签
    """
    raw_iter = pd.read_csv(data_filepath)

    # 提取标签，假设标签在最后一列
    labels = raw_iter.iloc[:, -1].values  # 更简洁的标签提取方式

    # 加载BERT嵌入、主题分布、手工特征
    bert_embedding = np.load(bert_embedding_filepath)
    topic_dists = np.load(document_topics_filepath)
    manual_features = np.load(manual_features_filepath)

    # 合并特征
    features = np.concatenate((bert_embedding, topic_dists, manual_features), axis=1)

    return features, labels

def load_data_for_ml(config):
    """
    根据配置加载训练和测试数据。
    """
    train_features, train_labels = data_process_for_ml(
        config.train_data_filepath,
        config.train_bert_embedding_filepath,
        config.train_document_topics_filepath,
        config.train_manual_features_filepath
    )

    test_features, test_labels = data_process_for_ml(
        config.test_data_filepath,
        config.test_bert_embedding_filepath,
        config.test_document_topics_filepath,
        config.test_manual_features_filepath
    )

    # 转换为 torch.Tensor 形式
    train_features, test_features = torch.tensor(train_features, dtype=torch.float32), torch.tensor(test_features, dtype=torch.float32)
    train_labels, test_labels = torch.tensor(train_labels, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32)

    return train_features, test_features, train_labels, test_labels

class LogisticRegression(nn.Module):
    """
    逻辑回归模型

    参数:
    - input_dim: 输入特征的维度
    - num_class: 类别数（默认为2，适用于二分类任务）
    """
    def __init__(self, input_dim, num_class=2):
        super(LogisticRegression, self).__init__()

        # 分类层：线性映射，将输入特征映射到类别数
        self.classifier = nn.Linear(input_dim, num_class)

    def forward(self, inputs):
        """
        前向传播逻辑
        """
        logits = self.classifier(inputs)  # 输出未激活的 logits
        return logits  # 不再使用 Sigmoid，直接返回 logits

# 用法示例
if __name__ == '__main__':
    # 假设有一个配置对象 config，包含了必要的文件路径
    train_features, test_features, train_labels, test_labels = load_data_for_ml(config)

    # 初始化模型
    model = LogisticRegression(input_dim=train_features.shape[1], num_class=1)

    # 损失函数使用 BCEWithLogitsLoss 处理二分类任务
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 示例的训练循环
    for epoch in range(10):
        model.train()

        # 前向传播
        logits = model(train_features)
        loss = criterion(logits.squeeze(), train_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 评估部分可以加上准确率或其他指标的计算
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
