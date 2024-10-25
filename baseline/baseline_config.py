import os
import torch
import json


class Config:
    """此类用于定义超参数"""
    def __init__(self, config_file=None):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))  # 获取Config类所在脚本的完整路径
        self.dataset_dir = os.path.join(self.project_dir, 'data')  # 数据集文件夹
        self.train_data_filepath = os.path.join(self.dataset_dir, 'train.csv')
        self.val_data_filepath = os.path.join(self.dataset_dir, 'test.csv')
        self.test_data_filepath = os.path.join(self.dataset_dir, 'test.csv')
        self.model_save_dir = os.path.join(self.project_dir, 'result')
        self.user_dict_filepath = os.path.join(self.dataset_dir, 'user_dict.txt')  # 自定义词典
        self.stopwords_filepath = os.path.join(self.dataset_dir, 'stopwords.txt')  # 停用词
        self.segmented_data_save_dir = os.path.join(self.dataset_dir, 'segmented_cache')
        self.segmented_train_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_train.txt')
        self.segmented_val_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')
        self.segmented_test_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')

        self.upper_dir = os.path.dirname(self.project_dir)
        self.train_bert_embedding_filepath = os.path.join(self.upper_dir, 'cache', 'bert_embedding', 'train_bert_embedding.npy')
        self.test_bert_embedding_filepath = os.path.join(self.upper_dir, 'cache', 'bert_embedding', 'test_bert_embedding.npy')
        self.train_document_topics_filepath = os.path.join(self.upper_dir, 'cache', 'train_lda_embedding', 'train_lda_embedding_21.npy')
        self.val_document_topics_filepath = os.path.join(self.upper_dir, 'cache', 'train_lda_embedding', 'test_lda_embedding_21.npy')
        self.test_document_topics_filepath = os.path.join(self.upper_dir, 'cache', 'train_lda_embedding', 'test_lda_embedding_21.npy')
        self.train_manual_features_filepath = os.path.join(self.upper_dir, 'cache', 'manual_feature', 'train_manual_feature.npy')
        self.val_manual_features_filepath = os.path.join(self.upper_dir, 'cache', 'manual_feature', 'test_manual_feature.npy')
        self.test_manual_features_filepath = os.path.join(self.upper_dir, 'cache', 'manual_feature', 'test_manual_feature.npy')
        self.word2vec_save_dir = os.path.join(self.model_save_dir, 'word2vec')

        self.tokenize = True  # 是否以分词的形式进行词典映射
        self.tokenize_type = 'ltp'  # ltp or jieba

        # 确保必要的文件夹存在
        file_dir = [self.model_save_dir, self.segmented_data_save_dir]
        for dir_ in file_dir:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # 设备选择
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 一些超参数
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 1e-5
        self.embedding_dim = 300
        self.num_class = 2
        self.hidden_dim = 512
        self.topic_dim = 21  # lda主题个数
        self.manual_features_dim = 2
        self.vector_dim = 300  # word2vec生成词向量维度
        self.filter_size = 3  # 卷积核宽度
        self.num_filter = 128  # 卷积核个数
        self.num_head = 8
        self.num_layers = 1

        # 如果传递了配置文件，就用配置文件覆盖默认值
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        """从 json 文件加载配置"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def save_config(self, save_path):
        """保存配置为 json 文件"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)


if __name__ == '__main__':
    config = Config()
    print(config.dataset_dir)

    # 加载 JSON 配置文件
    # config = Config(config_file='config.json')
    # config.save_config('new_config.json')
