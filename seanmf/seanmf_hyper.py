import os
import argparse
import numpy as np
from seanmf.utils import read_docs, read_vocab, calculate_PMI
from seanmf.seanmf_model import SeaNMFL1

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--model', default='seanmf', help='nmf | seanmf')
parser.add_argument('--max_iter', type=int, default=500, help='max number of iterations')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--beta', type=float, default=0.0, help='beta')
parser.add_argument('--max_err', type=float, default=0.1, help='stop criterion')
parser.add_argument('--fix_seed', type=bool, default=True, help='set random seed 0')
args = parser.parse_args()

# 超参数调优范围
alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0]
beta_values = [0.1, 0.3, 0.5, 0.7, 1.0]

best_alpha = None
best_beta = None
PMI_max = float('-inf')
best_n_topics = None

# 读取文档和词汇
docs = read_docs(args.corpus_file)
vocab = read_vocab(args.vocab_file)
n_docs = len(docs)
n_terms = len(vocab)
print(f'n_docs={n_docs}, n_terms={n_terms}')

# 检查临时文件夹
tmp_folder = 'seanmf'
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

# 计算共现矩阵
print('Calculating co-occurrence matrix...')
dt_mat = np.zeros([n_terms, n_terms])
for itm in docs:
    for kk in itm:
        for jj in itm:
            dt_mat[int(kk), int(jj)] += 1.0
print('Co-occurrence matrix done')
print('-' * 50)

# 计算 PPMI 矩阵
print('Calculating PPMI matrix...')
D1 = np.sum(dt_mat)
SS = D1 * dt_mat
for k in range(n_terms):
    SS[k] /= np.sum(dt_mat[k])
for k in range(n_terms):
    SS[:, k] /= np.sum(dt_mat[:, k])
dt_mat = None  # 释放内存
SS[SS == 0] = 1.0
SS = np.log(SS)
SS[SS < 0.0] = 0.0
print('PPMI matrix done')
print('-' * 50)

# 读取文档-词汇矩阵
print('Reading term-document matrix...')
dt_mat = np.zeros([n_terms, n_docs])
for k in range(n_docs):
    for j in docs[k]:
        dt_mat[j, k] += 1.0
print('Term-document matrix done')
print('-' * 50)

# 超参数调优，使用不同的 alpha, beta 和主题数量
for n_topics in range(50, 221, 10):
    for alpha in alpha_values:
        for beta in beta_values:
            model = SeaNMFL1(
                dt_mat, SS,
                alpha=alpha,
                beta=beta,
                n_topic=n_topics,
                max_iter=args.max_iter,
                max_err=args.max_err,
                fix_seed=args.fix_seed
            )

            # 训练模型并获取矩阵 W, Wc, H
            W, Wc, H = model.get_lowrank_matrix()
            n_topic = W.shape[1]

            # 计算每个主题的 PMI 值
            PMI_arr = []
            n_topKeyword = 10
            for k in range(n_topic):
                topKeywordsIndex = W[:, k].argsort()[::-1][:n_topKeyword]
                PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))

            avg_PMI = np.average(np.array(PMI_arr))
            print(f'Topic={n_topics}, Alpha={alpha}, Beta={beta}, Average PMI={avg_PMI}')

            # 保存最佳结果
            if avg_PMI > PMI_max:
                PMI_max = avg_PMI
                best_alpha = alpha
                best_beta = beta
                best_n_topics = n_topics

# 输出最佳超参数组合
print(f"Best alpha: {best_alpha}")
print(f"Best beta: {best_beta}")
print(f"Best number of topics: {best_n_topics}")
print(f"Max Average PMI: {PMI_max}")
