import re
import numpy as np

def read_docs(file_name):
    """
    从指定文件中读取文档。每个文档由词汇索引的列表表示。
    """
    print('Reading documents from:', file_name)
    print('-' * 50)

    docs = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        for line in fp:
            arr = re.split('\s+', line.strip())  # 使用strip移除换行符，\s+匹配多个空白字符
            if arr:  # 确保非空
                arr = [int(idx) for idx in arr if idx]  # 过滤空字符串并转换为整数
                docs.append(arr)

    return docs


def read_vocab(file_name):
    """
    从指定文件中读取词汇表。词汇表的每一行包含一个词汇。
    """
    print('Reading vocabulary from:', file_name)
    print('-' * 50)

    vocab = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        for line in fp:
            word = line.strip().split()[0]  # 获取每行的第一个词
            vocab.append(word)

    return vocab


def calculate_PMI(AA, topKeywordsIndex):
    """
    计算选定关键词之间的平均点互信息（PMI）。

    参数：
    AA -- 词语共现矩阵
    topKeywordsIndex -- 关键词索引列表

    返回：
    avg_PMI -- 选定关键词之间的平均PMI
    """
    D1 = np.sum(AA)  # 计算共现矩阵的总和
    n_tp = len(topKeywordsIndex)
    PMI = []

    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                # 如果关键词没有共现，PMI为0
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    # 计算关键词 index1 和 index2 的出现次数
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    # 计算PMI，log(P(index1,index2)/(P(index1)*P(index2)))
                    PMI_value = np.log(AA[index1, index2] * D1 / (C1 * C2))
                    PMI.append(PMI_value)

    # 计算平均PMI
    avg_PMI = 2.0 * np.sum(PMI) / (n_tp * (n_tp - 1.0))

    return avg_PMI
