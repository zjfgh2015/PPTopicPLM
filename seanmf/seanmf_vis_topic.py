import argparse
import numpy as np
from seanmf.utils import read_docs, read_vocab, calculate_PMI

def calculate_cooccurrence_matrix(docs, n_terms):
    """
    计算词语的共现矩阵。
    """
    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                if kk != jj:
                    dt_mat[int(kk), int(jj)] += 1.0
    print('Co-occurrence matrix computation done.')
    return dt_mat

def visualize_topics(W, dt_mat, vocab, n_topKeyword=10):
    """
    可视化主题，计算每个主题的关键词和平均PMI。
    """
    n_topic = W.shape[1]
    print(f'n_topic={n_topic}')

    PMI_arr = []
    for k in range(n_topic):
        topKeywordsIndex = W[:, k].argsort()[::-1][:n_topKeyword]
        pmi = calculate_PMI(dt_mat, topKeywordsIndex)
        PMI_arr.append(pmi)

    avg_pmi = np.average(np.array(PMI_arr))
    print(f'Average PMI={avg_pmi}')

    index = np.argsort(PMI_arr)  # 按照PMI从小到大排序
    for k in index:
        print(f'Topic {k+1}: PMI={PMI_arr[k]:.4f}', end=' ')
        top_words = [vocab[w] for w in np.argsort(W[:, k])[::-1][:n_topKeyword]]
        print(" ".join(top_words))

def main(args):
    # 读取文档和词汇表
    docs = read_docs(args.corpus_file)
    vocab = read_vocab(args.vocab_file)
    n_docs = len(docs)
    n_terms = len(vocab)
    print(f'n_docs={n_docs}, n_terms={n_terms}')

    # 计算词语共现矩阵
    dt_mat = calculate_cooccurrence_matrix(docs, n_terms)

    # 加载模型输出的W矩阵
    W = np.loadtxt(args.par_file, dtype=float)

    # 可视化主题
    visualize_topics(W, dt_mat, vocab, args.n_topKeyword)

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description="Visualize Topics from SeaNMF results.")
    parser.add_argument('--corpus_file', default='data/test_doc_term_mat.txt', help='term document matrix file')
    parser.add_argument('--vocab_file', default='data/test_vocab.txt', help='vocab file')
    parser.add_argument('--par_file', default='seanmf/W.txt', help='model results file')
    parser.add_argument('--n_topKeyword', type=int, default=10, help='number of top keywords to display per topic')
    args = parser.parse_args()

    main(args)
