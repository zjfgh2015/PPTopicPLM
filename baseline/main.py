import os
import torch
from torch import nn
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
from baseline.utils.data_process import FakeNewsDataset, CollateFn, save_csv
from baseline.utils.metrics import cal_precision, cal_recall, cal_f1
from baseline.baseline_config import Config
from baseline.model.text_cnn import TextCNN
from baseline.model.lstm import BiGRU
from baseline.model.word2vec import Word2vecLDA
from baseline.model.machine_learning_methods import LogisticRegression
from baseline.model.transformer import Transformer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def save_text(filename, content):
    with open(filename, 'a+', encoding='utf-8') as outputs:
        for line in content:
            outputs.write(str(line)+'\n')


def get_model(model_name, config, data_loader):
    """
    根据模型名称和配置获取模型实例。
    """
    if model_name == 'TextCNN':
        return TextCNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                       filter_size=config.filter_size, num_filter=config.num_filter, num_class=config.num_class)

    elif model_name == 'Bi-GRU':
        return BiGRU(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                     hidden_dim=config.hidden_dim, num_class=config.num_class)

    elif model_name == 'Word2vecLDA':
        return Word2vecLDA(input_dim=config.vector_dim + config.topic_dim,
                           hidden_dim=config.hidden_dim, num_class=config.num_class)

    elif model_name == 'LogisticRegression':
        return LogisticRegression(input_dim=config.topic_dim + config.manual_features_dim,
                                  num_class=config.num_class)

    elif model_name == 'Transformer':
        return Transformer(vocab_size=data_loader.vocab_size, num_class=config.num_class, d_model=config.embedding_dim,
                           dim_feedforward=config.hidden_dim, num_head=config.num_head, num_layers=config.num_layers)

    else:
        raise ValueError(f'模型名称输入有误：{model_name}')


def train(config, model_name=None):
    logging.info(f"Starting training for model: {model_name}")

    # 数据加载
    data_loader = FakeNewsDataset(config, model_name=model_name)
    train_iter, val_iter = data_loader.load_data(only_test=False, collate_fn=CollateFn.get_collate_fn(model_name),
                                                 model_name=model_name)

    # 模型获取与初始化
    model = get_model(model_name, config, data_loader).to(config.device)
    model_save_path = os.path.join(config.model_save_dir, f'mode_{model_name}.pt')

    # 定义损失函数、优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    model.train()

    # 训练
    max_f1 = 0
    for epoch in range(config.epochs):
        total_loss = 0
        total_acc = 0
        labels = []

        for batch in tqdm(train_iter, desc=f'Training Epoch {epoch + 1}'):
            inputs, targets = [x.to(config.device) for x in batch]
            optimizer.zero_grad()

            # 前向传播
            probs = model(inputs)
            loss = criterion(probs, targets)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            total_loss += loss.item()
            total_acc += (probs.argmax(dim=1) == targets).sum().item()
            labels.extend(targets.tolist())

        avg_loss = total_loss / len(train_iter)
        avg_acc = total_acc / len(labels)

        # 验证
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(model, criterion, val_iter, config.device, model_name)

        # 记录日志
        logging.info(f'Epoch {epoch + 1}/{config.epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_p:.4f}, Recall: {val_r:.4f}, F1: {val_f1:.4f}')

        if val_f1 > max_f1:
            max_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path} with F1: {val_f1:.4f}")

        scheduler.step()


def evaluate(model, criterion, data_iter, device, model_name=None):
    """
    模型评估函数
    """
    test_acc = 0
    test_loss = 0
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(data_iter, desc='Evaluating'):
            inputs, targets = [x.to(device) for x in batch]
            probs = model(inputs)

            loss = criterion(probs, targets)
            test_loss += loss.item()
            test_acc += (probs.argmax(dim=1) == targets).sum().item()

            y_pred.extend(probs.argmax(dim=1).tolist())
            y_true.extend(targets.tolist())

    avg_loss = test_loss / len(data_iter)
    avg_acc = test_acc / len(y_true)
    p = cal_precision(y_true, y_pred)
    r = cal_recall(y_true, y_pred)
    f1 = cal_f1(y_true, y_pred)

    return avg_loss, avg_acc, p, r, f1, y_true, y_pred
