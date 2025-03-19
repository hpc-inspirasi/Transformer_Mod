import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
INPUT_DIM = 1
MODEL_DIM = 64
N_HEADS = 4
N_LAYERS = 3
SEQ_LENGTH = 30
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev) 

# Generate synthetic time series data
np.random.seed(42)
time_steps = 2000

# Normal pattern (sinusoidal signal)
x = np.arange(0, time_steps)
normal_data = np.sin(0.02 * x) + 0.1 * np.random.randn(time_steps)

# Introduce anomalies
anomaly_indices = np.random.choice(time_steps, size=20, replace=False)
anomaly_data = normal_data.copy()
anomaly_data[anomaly_indices] += np.random.uniform(2, 4, size=len(anomaly_indices))

# Plotting the data
plt.figure(figsize=(15, 5))
plt.plot(normal_data, label='Normal Data')
plt.plot(anomaly_data, label='Anomalous Data')
plt.title('Synthetic Time Series Data with Anomalies')
plt.legend()
#plt.show()

# Save data to a DataFrame
data = pd.DataFrame({
    "value": anomaly_data
})

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, n_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x
# Prepare data
def prepare_data(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output[:, -1], y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

X, y = prepare_data(data["value"].values, SEQ_LENGTH)
X_train, y_train = X[:-400], y[:-400]
X_test, y_test = X[-400:], y[-400:]

# Convert to tensors
train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float().unsqueeze(-1), torch.tensor(y_train).float().unsqueeze(-1)),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float().unsqueeze(-1), torch.tensor(y_test).float().unsqueeze(-1)),
                         batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
model = TransformerModel(INPUT_DIM, MODEL_DIM, N_HEADS, N_LAYERS)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}")

