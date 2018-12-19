
import torch 
import torch.nn as nn
import torch.utils.data as utils

class Network(nn.Module):
    
    def __init__(self,device, input_size, hidden_size, nb_layers, dropout, nb_classes=2):
        super(Network, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm0 = nn.LSTM(input_size, hidden_size, nb_layers, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size//2, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size//2, nb_classes)
        self.activation = nn.ReLU()
        
    
    def forward(self, x):
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.nb_layers, x.size(0), self.hidden_size).to(self.device)
        
        h1 = torch.zeros(self.nb_layers, x.size(0), self.hidden_size//2).to(self.device)
        c1 = torch.zeros(self.nb_layers, x.size(0), self.hidden_size//2).to(self.device)
        
        # Forward propagate LSTM
        self.lstm0.flatten_parameters()
        out, _ = self.lstm0(x, (h0, c0))
        out = self.activation(out)
        out = self.dropout(out)
        
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(out, (h1, c1))
        out = self.activation(out)
        
        # retrieve hidden state of the last time step
        out = self.fc(out[:, -1, :])
       
        return out