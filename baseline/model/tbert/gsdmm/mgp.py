import numpy as np
from numpy.random import multinomial
from numpy import log, exp, argmax
import json
from tqdm.auto import tqdm  # 用于进度条

class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for _ in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        return np.argmax(multinomial(1, p))

    def fit(self, docs, vocab_size):
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        d_z = [None for _ in range(len(docs))]

        # 初始化
        for i, doc in enumerate(docs):
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        # Gibbs采样过程
        for _iter in tqdm(range(n_iters), desc="Gibbs Sampling"):
            total_transfers = 0

            for i, doc in enumerate(docs):
                z_old = d_z[i]
                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                p = self.score(doc)
                z_new = self._sample(p)

                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print(f"Iteration {_iter}: {total_transfers} documents transferred, {cluster_count_new} clusters populated")

            if total_transfers == 0 and _iter > 25:
                print("Converged. Breaking out.")
                break

        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = np.zeros(K)

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = sum(log(n_z_w[label].get(word, 0) + beta) for word in doc)
            lD2 = sum(log(n_z[label] + V * beta + j - 1) for j in range(1, doc_size + 1))
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        p /= np.sum(p)
        return p

    def choose_best_label(self, doc):
        p = self.score(doc)
        return argmax(p), max(p)
