
import torch 
import torch.nn as nn
import torch.utils.data as utils

class Network(nn.Module):
    
    def __init__(self,device, batch_size,input_size, hidden_size, nb_layers, dropout, nb_classes=2):
        super(Network, self).__init__()
        
        self.device = device
        
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm0 = nn.LSTM(input_size, hidden_size, nb_layers, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size//2, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size//2, nb_classes)
        self.activation = nn.ReLU()
        self.hidden_0 = self.init_hidden(batch_size=batch_size,
                                         denominator = 1)
        self.hidden_1 = self.init_hidden(batch_size=batch_size,
                                         denominator = 2)
        
    def init_hidden(self, batch_size,denominator=1):
        h = torch.zeros(self.nb_layers,batch_size,
                        self.hidden_size//denominator,device = self.device)
        c = torch.zeros(self.nb_layers, batch_size,
                        self.hidden_size//denominator,device = self.device)
        return(h,c)
    
    def forward(self, x):
        
        
        # Forward propagate LSTM
        self.lstm0.flatten_parameters()
        out, _ = self.lstm0(x, self.hidden_0)
        out = self.activation(out)
        out = self.dropout(out)
        
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(out, self.hidden_1)
        out = self.activation(out)
        
        # retrieve hidden state of the last time step
        out = self.fc(out[:, -1, :])
       
        return out