import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from gensim import corpora
import gensim
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def lda_preprocess(data_tokenized, id2word=None):
    if id2word is None:
        id2word = corpora.Dictionary(data_tokenized)

    processed_texts = data_tokenized
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    return corpus, id2word, processed_texts


def generate_topic_distributions(data_tokenized, lda_model, id2word):
    topic_distributions = []
    for tokens in data_tokenized:
        bow = id2word.doc2bow(tokens)
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0, minimum_phi_value=None)
        topic_vector = [topic[1] for topic in topic_distribution]
        topic_distributions.append(topic_vector)

    topic_distributions = torch.tensor(topic_distributions)
    return topic_distributions

# Define the tBERT model
class TBERTModel(nn.Module):
    def __init__(self, num_topics):
        super(TBERTModel, self).__init__()
        self.hidden_layer = nn.Linear(848, 768)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, bert_embedding,topic_distribution):
        combined_features = torch.cat([bert_embedding, topic_distribution], dim=1)
        hidden = self.hidden_layer(combined_features)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        probabilities = self.softmax(logits)
        return probabilities


# Set hyperparameters
num_topics = 80
num_epochs = 10
learning_rate = 2e-5

# Load the embedding vectors
train_embeddings = torch.from_numpy(np.load("train_embeddings4.npy")).float()
test_embeddings = torch.from_numpy(np.load("test_embeddings4.npy")).float()

# Load the labels
train_labels = torch.tensor(pd.read_csv("data/train.csv", encoding='utf-8')["label"])
test_labels = torch.tensor(pd.read_csv("data/test.csv", encoding='utf-8')["label"])

# Define BERT model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained('./final_model4')

# Load the texts
train_sentences = pd.read_csv("data/train.csv", encoding='utf-8')["text"].tolist()
test_sentences = pd.read_csv("data/test.csv", encoding='utf-8')["text"].tolist()

# Create tBERT model instance
tbert_model = TBERTModel( num_topics)
optimizer = torch.optim.AdamW(tbert_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Perform LDA topic modeling
train_tokenized = [tokenizer.tokenize(sentence) for sentence in train_sentences]
test_tokenized = [tokenizer.tokenize(sentence) for sentence in test_sentences]
corpus, id2word, _ = lda_preprocess(train_tokenized)
corpus1, id2word1, _ = lda_preprocess(test_tokenized)

# Train LDA model
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10)
test_lda_model = gensim.models.LdaModel(corpus=corpus1, id2word=id2word1, num_topics=num_topics, passes=10)

# Training loop
for epoch in range(num_epochs):
    tbert_model.train()
    optimizer.zero_grad()

    # Generate topic distributions for train sentences using LDA model
    train_topic_distributions = generate_topic_distributions(train_tokenized, lda_model, id2word)

    logits = tbert_model(train_embeddings,train_topic_distributions)
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

    tbert_model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():

        # Generate topic distributions for test sentences using LDA model
        test_topic_distributions = generate_topic_distributions(test_tokenized, test_lda_model, id2word1)

        logits = tbert_model(test_embeddings,test_topic_distributions)
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

