import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_path = "dataset/MSL/MSL_train_2500.npy"
test_path = "dataset/MSL/MSL_test_2500.npy"
labels_path = "dataset/MSL/MSL_test_label.npy"

train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)
labels = np.load(labels_path)

# Normalize data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

print(f"Train Data shape: {train_data.shape[1]}")
print(f"Test Data shape: {test_data.shape[1]}")

# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

# Create sequences for Transformer model
def create_sequences(data, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i+seq_length])
    return torch.stack(sequences).to(device), torch.tensor(seq_labels).to(device)

seq_length = 50 # Window size for time series
X_train, y_train = create_sequences(train_tensor, labels_tensor, seq_length)
X_test, y_test = create_sequences(test_tensor, labels_tensor, seq_length)

# Create DataLoader
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Transformer Model for Anomaly Detection
class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[:, -1, :]) # Use last output token
        return x.squeeze()

# Transformer Model for Trend Detection
class TransformerTrendDetector(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)
 
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[:, -1, :]) # Use last output token
        return x.squeeze()

# Initialize models
num_models = 3 # Number of models in the ensemble
anomaly_models = [TransformerAnomalyDetector(input_dim=train_data.shape[1], num_heads=5).to(device) for _ in range(num_models)]
trend_models = [TransformerTrendDetector(input_dim=train_data.shape[1], num_heads=5).to(device) for _ in range(num_models)]
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(anomaly_models[0].parameters(), lr=0.001)
writer = SummaryWriter(log_dir='runs/anomaly_detection')

# Check if saved model exists
save_path = "checkpoints/transformer_anomaly_detector.pth"
if os.path.exists(save_path):
    print(f"Loading saved model from {save_path}")
    for model in anomaly_models:
        model.load_state_dict(torch.load(save_path))
else:
    print("No saved model found, starting training from scratch.")

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader_train:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        losses = []
    for model in anomaly_models:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        losses.append(loss)
        total_loss += torch.mean(torch.stack(losses)).item() # Average loss across all models

    for loss in losses:
        loss.backward()
        optimizer.step()
    writer.add_scalar('Loss/train', total_loss / len(dataloader_train), epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader_train):.4f}")
writer.close()

# Save model
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(anomaly_models[0].state_dict(), save_path)

# Check if model is saved
if os.path.exists(save_path):
    print(f"Model successfully saved at {save_path}")
else:
    print("Error: Model save failed!")

# Evaluation on test data (Ensemble using Hard Voting)
for model in anomaly_models:
    model.eval()

predictions, ground_truths = [], []
reconstructed_signals = []

with torch.no_grad():
    for batch_X, batch_y in dataloader_test:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        model_outputs = [model(batch_X) for model in anomaly_models]
        reconstructed_signals.extend(reconstructed.cpu().numpy())
        # Hard voting: Get majority vote from the model outputs
        votes = [torch.sign(output).cpu().numpy() for output in model_outputs]
        majority_vote = np.sign(np.sum(votes, axis=0)) # Majority vote: +1 for anomaly, -1 for normal
        predictions.extend(majority_vote)
        ground_truths.extend(batch_y.cpu().numpy())

    # Compute evaluation metrics
    threshold = 0.5 # Default threshold for binary classification
    pred_labels = (np.array(predictions) > threshold).astype(int)
    y_true = np.array(ground_truths)
    accuracy = accuracy_score(y_true, pred_labels)
    auc_score = roc_auc_score(y_true, predictions)
    f1 = f1_score(y_true, pred_labels)
    precision = precision_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels)
    print(f"Test Accuracy: {accuracy:.4f}, AUC Score: {auc_score:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_signals, label="Reconstructed Signal", color="green", alpha=0.7)
plt.plot(y_true, label="True Anomalies", color="red")
plt.plot(predictions, label="Predicted Anomalies (Ensemble - Voting)", color="blue", alpha=0.7)
plt.title("Anomaly Detection Results on MSL Dataset (Ensemble - Hard Voting)")
plt.legend()
plt.show()
