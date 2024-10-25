import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from gensim.models import Word2Vec
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# 配置类，包含文件路径和模型参数
class Config:
    def __init__(self):
        self.vector_dim = 300
        self.word2vec_save_dir = './word2vec'
        self.train_data_filepath = './data/train.csv'
        self.test_data_filepath = './data/test.csv'
        self.user_dict_filepath = './data/user_dict.txt'
        self.segmented_train_filepath = './data/segmented_train.txt'
        self.segmented_test_filepath = './data/segmented_test.txt'
        self.tokenize_type = 'ltp'


def train_save_word2vec_model(config, sentences, save_model=False):
    """训练Word2Vec模型，并根据需求保存模型"""
    word2vec_model = Word2Vec(sentences=sentences,
                              vector_size=config.vector_dim,
                              window=5,
                              sg=1,
                              hs=0,
                              negative=5,
                              min_count=2,
                              epochs=10,
                              workers=4,
                              seed=1234)
    if save_model:
        model_path = os.path.join(config.word2vec_save_dir, 'word2vec_model')
        print(f'Saving Word2Vec Model to {model_path}...')
        word2vec_model.save(model_path)
    return word2vec_model


def load_word2vec_model(config):
    """加载保存的Word2Vec模型"""
    model_path = os.path.join(config.word2vec_save_dir, 'word2vec_model')
    print(f'Loading Word2Vec Model from {model_path}...')
    word2vec_model = Word2Vec.load(model_path)
    return word2vec_model


def infer_and_write_word2vec_vector(config, word2vec_model=None, max_vocab=None):
    """推断并保存词向量"""
    if word2vec_model is None:
        word2vec_model = load_word2vec_model(config)

    print('Inferring and writing Word2Vec vector...')
    vocab_len = len(word2vec_model.wv.index_to_key) if max_vocab is None else max_vocab
    word2vec_vector_file = os.path.join(config.word2vec_save_dir, 'word2vec_vector.log')

    with open(word2vec_vector_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(f'{vocab_len}\n')
        for i in tqdm(range(vocab_len), desc='Writing Word2Vec vectors'):
            word = word2vec_model.wv.index_to_key[i]
            vector_str = ' '.join(map(str, word2vec_model.wv[word]))
            line = f'{i} {word} {vector_str}\n'
            outfile.writelines(line)
    print(f'Word2Vec vectors saved to {word2vec_vector_file}')


def load_word2vec_vector(config, add_unk=True):
    """加载词向量"""
    vector_dim = config.vector_dim
    word2vec_vector_file = os.path.join(config.word2vec_save_dir, 'word2vec_vector.log')

    print(f"Reading Word2Vec vectors from {word2vec_vector_file}")
    word_vector_dict = {}
    vector_matrix = []

    if add_unk:
        word_vector_dict['<unk>'] = np.zeros(vector_dim)
        vector_matrix.append(np.zeros(vector_dim))

    with open(word2vec_vector_file, 'r', encoding='utf-8') as infile:
        next(infile)  # 跳过第一行
        for line in infile:
            line_data = line.rstrip().split(' ')
            word = line_data[1]
            vector = np.array([float(v) for v in line_data[2:]])
            word_vector_dict[word] = vector
            vector_matrix.append(vector)

    vector_matrix = np.array(vector_matrix)
    print(f'Loaded word vectors with shape: {vector_matrix.shape}')

    return word_vector_dict, vector_matrix


def word2vec_process(config):
    """处理数据并准备训练Word2Vec"""
    train_file_path = config.train_data_filepath
    raw_iter = pd.read_csv(train_file_path)
    sentences = []

    # 假设你有一个基本的分词函数
    for raw in tqdm(raw_iter.values, desc='Data Processing'):
        text = raw[1]  # 文本列
        words = text.split()  # 这里假设是空格分隔的分词结果
        sentences.append(words)

    return sentences


class Word2vecLDA(nn.Module):
    """Word2Vec + LDA模型"""

    def __init__(self, input_dim=300, hidden_dim=512, num_class=2):
        super(Word2vecLDA, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, word_vector, topic_vector):
        inputs = torch.cat([word_vector, topic_vector], dim=1)
        x = self.relu(self.fc1(inputs))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs


def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    """训练模型"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for word_vector, topic_vector, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(word_vector, topic_vector)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')


if __name__ == '__main__':
    # 配置初始化
    wv_config = Config()

    # 1. 处理数据并训练Word2Vec模型
    train_data_tokenized = word2vec_process(wv_config)
    wv_model = train_save_word2vec_model(wv_config, sentences=train_data_tokenized, save_model=True)
    infer_and_write_word2vec_vector(wv_config, wv_model)

    # 2. 加载词向量
    word_vector_dict, word_vector_matrix = load_word2vec_vector(wv_config)

    # 假设你已经从LDA模型获得了 topic_vectors
    # 以下代码展示了如何使用Word2Vec和LDA特征训练分类模型
    topic_vectors = np.random.rand(len(train_data_tokenized), wv_config.vector_dim)  # 假设有LDA生成的主题向量

    word_vectors = torch.tensor(word_vector_matrix, dtype=torch.float32)
    topic_vectors = torch.tensor(topic_vectors, dtype=torch.float32)

    labels = torch.randint(0, 2, (len(train_data_tokenized),))  # 假设二分类

    # 3. 构建DataLoader
    dataset = TensorDataset(word_vectors, topic_vectors, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. 初始化Word2Vec+LDA模型
    input_dim = word_vector_matrix.shape[1] + topic_vectors.shape[1]
    model = Word2vecLDA(input_dim=input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练模型
    train_model(model, train_loader, optimizer, criterion, num_epochs=10)
