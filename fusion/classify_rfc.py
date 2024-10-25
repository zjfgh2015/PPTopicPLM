from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the embedding vectors and H topic vectors
train_embeddings = np.load("embedding/train_embeddings.npy")
test_embeddings = np.load("embedding/test_embeddings.npy")

n_topics = 160  # Number of topics

# Load the labels
train_labels = pd.read_csv("data/train.csv", encoding='utf-8')["label"].values
test_labels = pd.read_csv("data/test.csv", encoding='utf-8')["label"].values

# Split the data into train and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(
    train_embeddings, train_labels, test_size=0.2, random_state=42)

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

# Convert data to PyTorch DataLoader for batch processing
batch_size = 64  # Set the batch size

# Loop through different topic sizes
precision_scores = []
accuracy_scores = []
f1_scores = []
recall_scores = []

for topic_id in range(10, n_topics + 1, 10):
    # Load H topic vectors
    train_H_topic = np.loadtxt(f"seanmf_results/new_train_H_{topic_id}.txt")
    test_H_topic = np.loadtxt(f"seanmf_results/new_test_H_{topic_id}.txt")

    # Concatenate the embedding vectors with H topic vectors
    train_data_concat = np.concatenate((train_H_topic[:5841, :], train_data), axis=1)
    validation_data_concat = np.concatenate((train_H_topic[-1461:], validation_data), axis=1)
    test_data_concat = np.concatenate((test_H_topic, test_embeddings), axis=1)

    # Standardize the data
    scaler = StandardScaler()
    train_data_concat = scaler.fit_transform(train_data_concat)
    validation_data_concat = scaler.transform(validation_data_concat)
    test_data_concat = scaler.transform(test_data_concat)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.from_numpy(train_data_concat).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    validation_data_tensor = torch.from_numpy(validation_data_concat).float()
    validation_labels_tensor = torch.from_numpy(validation_labels).long()
    test_data_tensor = torch.from_numpy(test_data_concat).float()

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = TensorDataset(validation_data_tensor, validation_labels_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    model = nn.Sequential(
        nn.Linear(train_data_tensor.shape[1], 128),
        nn.Tanh(),
        ResidualBlock(128, 64),  # Add Residual Block
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop
    num_epochs = 50
    patience = 5  # Early stopping patience
    best_f1_score = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_predictions = []
            val_targets = []
            for val_data, val_labels in validation_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_outputs = model(val_data).argmax(dim=1)
                val_predictions.extend(val_outputs.cpu().numpy())
                val_targets.extend(val_labels.cpu().numpy())

        val_f1_score = f1_score(val_targets, val_predictions)
        print(f"Validation F1 Score: {val_f1_score:.4f}")

        # Early stopping
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Test the model on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor.to(device)).argmax(dim=1).cpu().numpy()

    precision_scores.append(precision_score(test_labels, test_outputs))
    accuracy_scores.append(accuracy_score(test_labels, test_outputs))
    f1_scores.append(f1_score(test_labels, test_outputs))
    recall_scores.append(recall_score(test_labels, test_outputs))

# Print evaluation metrics for different topic numbers
for topic_id, precision, accuracy, recall, f1 in zip(range(10, n_topics + 1, 10), precision_scores, accuracy_scores,
                                                     recall_scores, f1_scores):
    print(f"Topic Number: {topic_id}")
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("-" * 20)
