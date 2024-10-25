import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the embedding vectors and H topic vectors
train_embeddings = np.load("embedding/train_embeddings.npy")
test_embeddings = np.load("embedding/test_embeddings.npy")

n_topics = 160  # Number of topics

# Load the labels
train_labels = pd.read_csv("data/train.csv", encoding='utf-8')["label"]
test_labels = pd.read_csv("data/test.csv", encoding='utf-8')["label"]

# Calculate evaluation metrics for different topic numbers
precision_scores = []
accuracy_scores = []
f1_scores = []
recall_scores = []

for topic_id in range(10, n_topics + 1, 10):
    # Load the H topic vectors for both train and test data
    train_H_topic = np.loadtxt(f"seanmf_results/new_train_H_{topic_id}.txt")
    test_H_topic = np.loadtxt(f"seanmf_results/new_test_H_{topic_id}.txt")

    # Concatenate the embedding vectors with H topic vectors
    train_data = np.concatenate((train_H_topic, train_embeddings), axis=1)
    test_data = np.concatenate((test_H_topic, test_embeddings), axis=1)

    # Scale the features using StandardScaler (or you can use MinMaxScaler)
    scaler = StandardScaler()  # Or MinMaxScaler() for normalization
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    model.fit(train_data, train_labels)

    # Predict on the test data
    test_predictions = model.predict(test_data)

    # Calculate the evaluation metrics
    precision_scores.append(precision_score(test_labels, test_predictions))
    accuracy_scores.append(accuracy_score(test_labels, test_predictions))
    f1_scores.append(f1_score(test_labels, test_predictions))
    recall_scores.append(recall_score(test_labels, test_predictions))

# Print evaluation metrics for different topic numbers
for topic_id, precision, accuracy, recall, f1 in zip(range(10, n_topics + 1, 10), precision_scores, accuracy_scores, recall_scores, f1_scores):
    print(f"Topic Number: {topic_id}")
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("-" * 20)
