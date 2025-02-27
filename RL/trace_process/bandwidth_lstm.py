import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BandwidthLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, sequence_length=5):
        super(BandwidthLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.scaler = MinMaxScaler()
        self.throughput_history = []
        
        self.to(self.device)
    
    def update_history(self, throughput):
        self.throughput_history.append(throughput)
        if len(self.throughput_history) > self.sequence_length:
            self.throughput_history.pop(0)
    
    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
        
    def process_throughput(self, throughput):
        self.update_history(throughput)
        if len(self.throughput_history) < self.sequence_length:
            return throughput
            
        data = np.array(self.throughput_history).reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(data)
        x = torch.FloatTensor(normalized_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            processed = self(x)
            processed = processed.cpu().numpy()
            processed = self.scaler.inverse_transform(processed.reshape(-1, 1))
        return processed[0, 0]
