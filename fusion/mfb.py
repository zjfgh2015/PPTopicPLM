import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import numpy as np
import pandas as pd


# 定义 MFB 模型
class MFB(nn.Module):
    def __init__(self, topic_dim):
        super(MFB, self).__init__()
        self.JOINT_EMB_SIZE = 1000
        self.Linear_t_proj = nn.Linear(topic_dim, self.JOINT_EMB_SIZE)
        self.Linear_s_proj = nn.Linear(1024, self.JOINT_EMB_SIZE)
        self.Dropout_M = nn.Dropout(p=0.5)
        self.Linear_predict = nn.Linear(200, 2)

    def forward(self, x, y):
        mfb_t_proj = self.Linear_t_proj(x)
        mfb_s_proj = self.Linear_s_proj(y)

        mfb_st_eltwise = torch.mul(mfb_t_proj, mfb_s_proj)
        mfb_st_drop = self.Dropout_M(mfb_st_eltwise)
        mfb_st_resh = mfb_st_drop.view(x.shape[0], 1, 200, 5)
        mfb_st_sumpool = torch.sum(mfb_st_resh, 3, keepdim=True)
        mfb_out = torch.squeeze(mfb_st_sumpool)
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        prediction = self.Linear_predict(mfb_l2)

        return prediction


# 加载数据
def load_data():
    train_embeddings = np.load("embedding/train_embeddings3.npy")
    test_embeddings = np.load("embedding/test_embeddings3.npy")
    train_labels = pd.read_csv("data/train.csv", encoding='utf-8')["label"]
    test_labels = pd.read_csv("data/test.csv", encoding='utf-8')["label"]

    return train_embeddings, test_embeddings, train_labels, test_labels


# 分割训练和验证集
def split_train_validation(train_embeddings, train_labels):
    return train_test_split(train_embeddings, train_labels, test_size=0.2, random_state=42)


# 训练模型
def train_model(model, train_embeddings, train_topics, train_labels, validation_embeddings, validation_topics,
                validation_labels, num_epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    train_embeddings_tensor = torch.from_numpy(train_embeddings).float()
    train_topic_tensor = torch.from_numpy(train_topics).float()
    train_labels_tensor = torch.from_numpy(train_labels.values).long()

    validation_embeddings_tensor = torch.from_numpy(validation_embeddings).float()
    validation_topic_tensor = torch.from_numpy(validation_topics).float()

    best_f1_score = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_topic_tensor, train_embeddings_tensor)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(validation_topic_tensor, validation_embeddings_tensor).argmax(dim=1).cpu().numpy()
            val_f1_score = f1_score(validation_labels, val_predictions)

        # Early stopping机制
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_model_state = model.state_dict()
        else:
            break

    return best_model_state


# 评估模型
def evaluate_model(model, test_embeddings, test_topics, test_labels):
    test_embeddings_tensor = torch.from_numpy(test_embeddings).float()
    test_topic_tensor = torch.from_numpy(test_topics).float()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_topic_tensor, test_embeddings_tensor).argmax(dim=1).cpu().numpy()

    return test_predictions


# 主程序
if __name__ == "__main__":
    n_topics = 160
    train_embeddings, test_embeddings, train_labels, test_labels = load_data()

    # 存储评价指标
    precision_scores = []
    accuracy_scores = []
    f1_scores = []
    recall_scores = []

    # 分割训练和验证集
    train_data, validation_data, train_labels, validation_labels = split_train_validation(train_embeddings,
                                                                                          train_labels)

    for topic_id in range(10, n_topics + 1, 10):
        train_H_topic = np.loadtxt(f"deberta_seanmf_results/new_train_H_{topic_id}.txt")
        test_H_topic = np.loadtxt(f"deberta_seanmf_results/new_test_H_{topic_id}.txt")

        model = MFB(topic_dim=topic_id)

        # 训练模型
        best_model_state = train_model(
            model,
            train_data, train_H_topic[:len(train_data)], train_labels,
            validation_data, train_H_topic[len(train_data):], validation_labels
        )

        # 加载最佳模型状态
        model.load_state_dict(best_model_state)

        # 测试集评估
        test_predictions = evaluate_model(model, test_embeddings, test_H_topic, test_labels)

        precision_scores.append(precision_score(test_labels, test_predictions))
        accuracy_scores.append(accuracy_score(test_labels, test_predictions))
        f1_scores.append(f1_score(test_labels, test_predictions))
        recall_scores.append(recall_score(test_labels, test_predictions))

    # 打印每个主题数的评估结果
    for topic_id, precision, accuracy, recall, f1 in zip(range(10, n_topics + 1, 10), precision_scores, accuracy_scores,
                                                         recall_scores, f1_scores):
        print(f"Topic Number: {topic_id}")
        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("-" * 20)
