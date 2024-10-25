import argparse
import re
from collections import Counter

# 加载自定义工具函数
from seanmf.utils import *

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--text_file', default='data/data.txt', help='input text file')
parser.add_argument('--corpus_file', default='data/doc_term_mat.txt', help='term document matrix file')
parser.add_argument('--vocab_file', default='data/vocab.txt', help='vocab file')
parser.add_argument('--vocab_max_size', type=int, default=10000, help='maximum vocabulary size')
parser.add_argument('--vocab_min_count', type=int, default=3, help='minimum frequency of the words')
args = parser.parse_args()

# Step 1: 创建词汇表
print('Creating vocabulary...')
with open(args.text_file, 'r', encoding='utf-8') as fp:
    # 使用 Counter 统计单词出现次数
    vocab_counter = Counter(re.split(r'\s+', line.strip()) for line in fp)

# 过滤低频词并按频率排序
vocab_arr = [(word, count) for word, count in vocab_counter.items() if count >= args.vocab_min_count]
vocab_arr = sorted(vocab_arr, key=lambda x: x[1], reverse=True)[:args.vocab_max_size]  # 取前max_size个
vocab_arr.sort()  # 词汇表按字母排序

# Step 2: 将词汇表写入文件
with open(args.vocab_file, 'w', encoding='utf-8') as fout:
    for word, count in vocab_arr:
        fout.write(f'{word} {count}\n')

# 创建单词到ID的映射
vocab2id = {word: idx for idx, (word, _) in enumerate(vocab_arr)}

# Step 3: 创建文档-词汇矩阵 (Term Document Matrix)
print('Creating document term matrix...')
with open(args.text_file, 'r', encoding='utf-8') as fp, open(args.corpus_file, 'w', encoding='utf-8') as fout:
    for line in fp:
        words = re.split(r'\s+', line.strip())
        # 将单词转换为ID，如果单词在词汇表中
        word_ids = [str(vocab2id[word]) for word in words if word in vocab2id]
        fout.write(' '.join(word_ids) + '\n')

print('Process completed successfully!')
