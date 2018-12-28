
import torch 
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self,batch_size,input_size, 
                 hidden_size, nb_layers, dropout):
        super(Network, self).__init__()
        
        
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 
                           nb_layers, batch_first=True,
                           dropout = dropout)
        self.fc = nn.Linear(hidden_size, 2)
        self.activation = nn.ReLU()
        return (h,c)
    
    def forward(self, sequence):
        
        out,_ = self.rnn(sequence)
        out = self.activation(out)
        out = self.fc(out[:, -1, :])
        return out