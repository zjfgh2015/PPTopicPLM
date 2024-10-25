import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import numpy as np
import pandas as pd

class iAFF(nn.Module):
    def __init__(self, topic_dim, sentence_dim):
        super(iAFF, self).__init__()
        self.topic_linear = nn.Linear(topic_dim, sentence_dim)
        self.final_linear = nn.Linear(sentence_dim, 2)  # 输出层调整为2个神经元以适应CrossEntropyLoss
        self.sigmoid = nn.Sigmoid()

    def forward(self, topic_vectors, sentence_vectors):
        expanded_topic_vectors = self.topic_linear(topic_vectors)
        weights = self.sigmoid(expanded_topic_vectors)

        # 结合权重计算最终的输出
        output_vectors = expanded_topic_vectors * weights + sentence_vectors * (1 - weights)
        output_vectors = self.final_linear(output_vectors)

        return output_vectors  # 返回未激活的 logits

# 加载嵌入向量和 H 主题向量
train_embeddings = np.load("embedding/train_embeddings.npy")
test_embeddings = np.load("embedding/test_embeddings.npy")

n_topics = 160  # 主题数量

# 加载标签
train_labels = pd.read_csv("data/train.csv", encoding='utf-8')["label"].values
test_labels = pd.read_csv("data/test.csv", encoding='utf-8')["label"].values

# 将数据转换为 PyTorch 的 Tensors 并发送到设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_embeddings = torch.from_numpy(train_embeddings).float().to(device)
test_embeddings = torch.from_numpy(test_embeddings).float().to(device)
train_labels = torch.from_numpy(train_labels).long().to(device)
test_labels = torch.from_numpy(test_labels).long().to(device)

sentence_dim = train_embeddings.shape[1]

for num_topics in range(10, n_topics + 1, 10):
    train_H_topic = np.loadtxt(f"seanmf_results/new_train_H_{num_topics}.txt")
    test_H_topic = np.loadtxt(f"seanmf_results/new_test_H_{num_topics}.txt")

    # 转换为 PyTorch 张量
    train_topic_vectors = torch.from_numpy(train_H_topic).float().to(device)
    test_topic_vectors = torch.from_numpy(test_H_topic).float().to(device)

    # 初始化模型、损失函数和优化器
    model = iAFF(num_topics, sentence_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()  # 清除上一步的梯度
        outputs = model(train_topic_vectors, train_embeddings)
        loss = criterion(outputs, train_labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        if (epoch + 1) % 10 == 0:  # 每10个epoch输出一次损失
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_topic_vectors, test_embeddings).argmax(dim=1).cpu().numpy()

    # 计算评价指标
    precision = precision_score(test_labels.cpu(), test_outputs)
    accuracy = accuracy_score(test_labels.cpu(), test_outputs)
    f1 = f1_score(test_labels.cpu(), test_outputs)
    recall = recall_score(test_labels.cpu(), test_outputs)

    # 输出最佳主题数量和模型表现
    print(f"Topic Number: {num_topics}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 20)
