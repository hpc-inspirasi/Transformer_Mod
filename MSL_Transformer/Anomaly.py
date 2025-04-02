import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
msl_path = "dataset/MSL/MSL_test.npy"
labels_path = "dataset/MSL/MSL_test_label.npy"

data = np.load(msl_path)
labels = np.load(labels_path)

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(data)

print(f"Data shape: {data.shape[1]}")

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

# Create sequences for Transformer model
def create_sequences(data, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i+seq_length])
    return torch.stack(sequences).to(device), torch.tensor(seq_labels).to(device)

seq_length = 50  # Window size for time series
X, y = create_sequences(data_tensor, labels_tensor, seq_length)

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Transformer Model for Anomaly Detection
class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Use last output token
        return x.squeeze()

# Initialize model, loss function, and optimizer
model = TransformerAnomalyDetector(input_dim=data.shape[1], num_heads=5).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if saved model exists
save_path = "checkpoints/transformer_anomaly_detector.pth"
if os.path.exists(save_path):
    print(f"Loading saved model from {save_path}")
    model.load_state_dict(torch.load(save_path))
else:
    print("No saved model found, starting training from scratch.")

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)

# Check if model is saved
if os.path.exists(save_path):
    print(f"Model successfully saved at {save_path}")
else:
    print("Error: Model save failed!")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(labels[seq_length:], label="True Anomalies", color="red")
plt.title("Anomaly Labels in MSL Dataset")
plt.legend()
plt.show()
