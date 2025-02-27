import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Create dataset function
def create_dataset(data, k):
    X, y = [], []
    for i in range(len(data) - k):
        X.append(data[i:i+k])
        y.append(data[i+k])
    return np.array(X), np.array(y)

# Parameters
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
k = 5  # Use the past k data points

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model function
def train_model(X, y):
    epochs = 100
    for epoch in range(epochs):
        model.train()
        outputs = model(X)  # X should be 3D here (batch_size, seq_len, input_size)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Process multiple files
log_files = glob.glob('../../data/bandwidth/train/mine/103.49.160.132/*.log')
for file in log_files:
    print(f'Processing {file}')
    
    # Load and preprocess data
    data = pd.read_csv(file, sep=' ', header=None)
    throughputs = data[1].values
    scaler = MinMaxScaler()
    throughputs_normalized = scaler.fit_transform(throughputs.reshape(-1, 1))
    X, y = create_dataset(throughputs_normalized, k)
    
    # Convert to tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().view(-1, 1)  # Ensure y is 2D with shape (batch_size, output_size)
    X = X.view(X.size(0), X.size(1), 1)  # Shape: (batch_size, seq_len, input_size)

    # Train model
    train_model(X, y)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predicted = model(X)
        predicted = scaler.inverse_transform(predicted.numpy())
    
    # Save predictions to txt file
    output_file = file.replace('.log', '_predictions.txt')
    np.savetxt(output_file, predicted, fmt='%.4f')
    print(f'Saved predictions to {output_file}')
