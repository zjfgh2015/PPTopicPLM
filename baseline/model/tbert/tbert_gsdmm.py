from .gsdmm import MovieGroupProcess
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def lda_preprocess(data_tokenized, id2word=None):
    if id2word is None:
        id2word = corpora.Dictionary(data_tokenized)

    processed_texts = data_tokenized
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    return corpus, id2word, processed_texts

def extract_topic_from_gsdmm_prediction(dist_over_topic):
    '''
    Extracts topic vectors from prediction
    '''
    global_topics = np.array(dist_over_topic)
    return global_topics

def infer_gsdmm_topics(gsdmm_model, texts):
    '''
    Predicts topic distribution for tokenized sentences
    '''
    dist_over_topic = [gsdmm_model.score(t) for t in texts]  # probability distribution over topics
    global_topics = extract_topic_from_gsdmm_prediction(dist_over_topic)
    return global_topics

def train_save_gsdmm_model(processed_texts, id2word):
    '''
    Trains GSDMM topic model
    '''
    gsdmm_model = MovieGroupProcess(K=80, alpha=0.1, beta=0.1, n_iters=30)
    vocab = set(x for doc in processed_texts for x in doc)
    n_terms = len(vocab)
    gsdmm_model.fit(processed_texts, n_terms)
    return gsdmm_model

# Define the TBERT model
class TBERTModel(nn.Module):
    def __init__(self, num_topics):
        super(TBERTModel, self).__init__()
        self.hidden_layer = nn.Linear(768 + num_topics, 768)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 2)

    def forward(self, bert_embedding, topic_distribution):
        combined_features = torch.cat([bert_embedding, topic_distribution], dim=1)
        hidden = self.hidden_layer(combined_features)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits

# Hyperparameters
num_topics = 80
num_epochs = 10
learning_rate = 2e-5

# Load the embedding vectors
train_embeddings = torch.from_numpy(np.load("train_embeddings4.npy")).float()
test_embeddings = torch.from_numpy(np.load("test_embeddings4.npy")).float()

# Load the labels
train_labels = torch.tensor(pd.read_csv("data/train.csv", encoding='utf-8')["label"].values)
test_labels = torch.tensor(pd.read_csv("data/test.csv", encoding='utf-8')["label"].values)

# Define BERT model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained('./final_model4')

# Load the texts
train_sentences = pd.read_csv("data/train.csv", encoding='utf-8')["text"].tolist()
test_sentences = pd.read_csv("data/test.csv", encoding='utf-8')["text"].tolist()

# Preprocess the data
train_tokenized = [tokenizer.tokenize(sentence) for sentence in train_sentences]
test_tokenized = [tokenizer.tokenize(sentence) for sentence in test_sentences]

corpus, id2word, processed_texts = lda_preprocess(train_tokenized)
_, _, processed_texts1 = lda_preprocess(test_tokenized)

# Train GSDMM model on both train and test sets combined
gsdmm_model = train_save_gsdmm_model(processed_texts + processed_texts1, id2word)

# Create tBERT model instance
tbert_model = TBERTModel(num_topics)
optimizer = torch.optim.AdamW(tbert_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    tbert_model.train()
    optimizer.zero_grad()

    # Generate topic distributions for train sentences using GSDMM model
    train_topic_distributions = infer_gsdmm_topics(gsdmm_model, train_tokenized)

    logits = tbert_model(train_embeddings, torch.tensor(train_topic_distributions).float())
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

    # Evaluation
    tbert_model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        # Generate topic distributions for test sentences using GSDMM model
        test_topic_distributions = infer_gsdmm_topics(gsdmm_model, test_tokenized)

        logits = tbert_model(test_embeddings, torch.tensor(test_topic_distributions).float())
        _, predicted = torch.max(logits, dim=1)
        total_correct += (predicted == test_labels).sum().item()
        total_samples += test_labels.size(0)

    accuracy = total_correct / total_samples

    # Calculate precision, recall, and F1-score
    predicted_labels = predicted.cpu().numpy()
    true_labels = test_labels.cpu().numpy()

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
