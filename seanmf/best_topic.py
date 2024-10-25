import argparse
import numpy as np
import os
from seanmf.utils import read_docs, read_vocab, calculate_PMI  # 假设这些函数已在 seanmf.utils 中定义

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--model_dir', default='seanmf/', help='Directory containing model files')
opt = parser.parse_args()

# 读取文档和词汇表
docs = read_docs(opt.corpus_file)
vocab = read_vocab(opt.vocab_file)
n_docs = len(docs)
n_terms = len(vocab)
print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

# 计算词汇共现矩阵
dt_mat = np.zeros([n_terms, n_terms])
for itm in docs:
    for kk in itm:
        for jj in itm:
            if kk != jj:
                dt_mat[int(kk), int(jj)] += 1.0
print('Co-occurrence matrix calculation done')

# 初始化最佳 PMI 和主题数
PMI_max = float('-inf')
best_topic = 0

# 遍历不同的主题数
for topic in range(10, 221, 10):
    try:
        # 动态加载模型文件
        W_file = os.path.join(opt.model_dir, f'W_{topic}.txt')
        Wc_file = os.path.join(opt.model_dir, f'Wc_{topic}.txt')
        H_file = os.path.join(opt.model_dir, f'H_{topic}.txt')

        if not (os.path.exists(W_file) and os.path.exists(Wc_file) and os.path.exists(H_file)):
            raise FileNotFoundError(f"Model files for topic {topic} not found")

        W = np.loadtxt(W_file, dtype=float)
        n_topic = W.shape[1]
        print('n_topic={}'.format(n_topic))

        # 计算每个主题的 PMI 值
        PMI_arr = []
        n_topKeyword = 10
        for k in range(n_topic):
            topKeywordsIndex = W[:, k].argsort()[::-1][:n_topKeyword]
            PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))

        avg_PMI = np.average(np.array(PMI_arr))
        print('Topic={}, Average PMI={}'.format(topic, avg_PMI))

        # 更新最佳主题
        if avg_PMI > PMI_max:
            PMI_max = avg_PMI
            best_topic = topic

    except Exception as e:
        print(f'Error occurred during processing topic {topic}. Error message: {str(e)}')

# 输出最佳主题数及其平均 PMI
print(f'Best Topic: {best_topic}, Max Average PMI: {PMI_max}')
