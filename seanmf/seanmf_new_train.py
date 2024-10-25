import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from seanmf.utils import read_docs, read_vocab
from seanmf.seanmf_model import SeaNMFL1


def encode_documents(docs, tokenizer, model):
    """
    将文档编码为向量表示，使用预训练的 Transformer 模型（如 RoBERTa）。
    """
    doc_vectors = []
    for doc in docs:
        inputs = tokenizer([doc], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        doc_vectors.append(vector)
    return doc_vectors


def calculate_cooccurrence_matrix(doc_vectors, n_terms):
    """
    计算词语的共现矩阵。
    """
    cooc_matrix = np.zeros([n_terms, n_terms])
    for doc_vector in doc_vectors:
        similarity_matrix = cosine_similarity(doc_vector.reshape(1, -1), doc_vector.reshape(1, -1))
        cooc_matrix += similarity_matrix
    return cooc_matrix


def calculate_ppmi_matrix(dt_mat, n_terms):
    """
    计算PPMI矩阵。
    """
    D1 = np.sum(dt_mat)
    SS = D1 * dt_mat.copy()

    for k in range(n_terms):
        SS[k] /= np.sum(dt_mat[k])
    for k in range(n_terms):
        SS[:, k] /= np.sum(dt_mat[:, k])

    SS[SS == 0] = 1.0
    SS = np.log(SS)
    SS[SS < 0.0] = 0.0
    return SS


def train_seanmf_model(n_topics, dt_mat, cooc_matrix, args, tmp_folder):
    """
    训练 SeaNMF 模型并保存结果。
    """
    model = SeaNMFL1(
        dt_mat, cooc_matrix,
        alpha=args.alpha,
        beta=args.beta,
        n_topic=n_topics,
        max_iter=args.max_iter,
        max_err=args.max_err,
        fix_seed=args.fix_seed
    )

    model.save_format(
        W1file=os.path.join(tmp_folder, f'new_W_{n_topics}.txt'),
        W2file=os.path.join(tmp_folder, f'new_Wc_{n_topics}.txt'),
        Hfile=os.path.join(tmp_folder, f'new_H_{n_topics}.txt')
    )


def main(args):
    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('./roberta')
    model = AutoModel.from_pretrained('./roberta')

    # 从数据文件中读取文档
    with open('data/data.txt', 'r', encoding='utf-8') as fp:
        docs = [doc.strip() for doc in fp.readlines()]

    dt = read_docs(args.corpus_file)
    vocab = read_vocab(args.vocab_file)
    n_dt = len(dt)
    n_terms = len(vocab)
    print(f'n_dt={n_dt}, n_terms={n_terms}')

    # 编码文档
    doc_vectors = encode_documents(docs, tokenizer, model)

    # 计算共现矩阵
    cooc_matrix = calculate_cooccurrence_matrix(doc_vectors, n_terms)
    print(f'共现矩阵计算完成, 形状: {cooc_matrix.shape}')

    # 创建临时目录
    tmp_folder = 'seanmf'
    os.makedirs(tmp_folder, exist_ok=True)

    # 如果选择 SeaNMF 模型
    if args.model.lower() == 'seanmf':
        print('计算词语共现矩阵和PPMI矩阵...')
        # 计算 PPMI 矩阵
        dt_mat = np.zeros([n_terms, n_terms])
        for itm in dt:
            for kk in itm:
                for jj in itm:
                    dt_mat[int(kk), int(jj)] += 1.0

        ppmi_matrix = calculate_ppmi_matrix(dt_mat, n_terms)
        print('PPMI 矩阵计算完成')

        print('读取 term-doc 矩阵...')
        dt_mat = np.zeros([n_terms, n_dt])
        for k in range(n_dt):
            for j in dt[k]:
                dt_mat[j, k] += 1.0

        print('term-doc 矩阵计算完成')

        # 训练多个主题数的 SeaNMF 模型
        for n_topics in range(10, args.n_topics + 1, 10):
            train_seanmf_model(n_topics, dt_mat, cooc_matrix, args, tmp_folder)

        print('SeaNMF 模型训练完成')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
    parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
    parser.add_argument('--model', default='seanmf', help='nmf | seanmf')
    parser.add_argument('--max_iter', type=int, default=500, help='max number of iterations')
    parser.add_argument('--n_topics', type=int, default=100, help='number of topics')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--beta', type=float, default=0.0, help='beta')
    parser.add_argument('--max_err', type=float, default=0.1, help='stop criterion')
    parser.add_argument('--fix_seed', type=bool, default=True, help='set random seed 0')
    args = parser.parse_args()

    main(args)
