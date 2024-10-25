from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import torch

# 自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']

        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        return {'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': torch.tensor(label, dtype=torch.long)}

# 1. 加载数据
def load_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None
    return train_df, test_df

# 2. 加载模型和分词器
def load_model_and_tokenizer(model_name, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# 3. 模型训练
def train_model(train_dataset, test_dataset, model, tokenizer, output_dir='./results', epochs=5, batch_size=4, learning_rate=2e-5):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    # 保存模型和分词器
    tokenizer.save_pretrained('tokenizer_folder')
    trainer.save_model("final_model")

    return trainer

# 4. 获取嵌入向量
def get_embeddings(dataset, model, trainer):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            input_ids = dataset[i]['input_ids'].unsqueeze(0).to(trainer.args.device)
            attention_mask = dataset[i]['attention_mask'].unsqueeze(0).to(trainer.args.device)
            embedding = model.bert(input_ids=input_ids, attention_mask=attention_mask)[1].cpu().numpy()
            embeddings.append(embedding)
    return np.concatenate(embeddings)

# 5. 保存嵌入向量
def save_embeddings(train_embeddings, test_embeddings):
    np.save('train_embeddings.npy', train_embeddings)
    np.save('test_embeddings.npy', test_embeddings)
    print("Embeddings saved successfully.")

# 主函数
if __name__ == "__main__":
    # 模型名称及文件路径
    model_name = "hfl/chinese-roberta-wwm-ext"
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    # 加载数据
    train_df, test_df = load_data(train_path, test_path)
    if train_df is None or test_df is None:
        print("Data loading failed.")
        exit()

    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer(model_name)

    # 定义数据集
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)

    # 训练模型
    trainer = train_model(train_dataset, test_dataset, model, tokenizer)

    # 获取训练集和测试集的嵌入向量
    train_embeddings = get_embeddings(train_dataset, model, trainer)
    test_embeddings = get_embeddings(test_dataset, model, trainer)

    # 保存嵌入向量
    save_embeddings(train_embeddings, test_embeddings)

    print("Process complete.")
