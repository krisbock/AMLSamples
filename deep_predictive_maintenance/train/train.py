

import torch 
import torch.nn as nn
import torch.utils.data as utils
from azureml.core import Run
import numpy as np
import pandas as pd
from utils import tensorize,to_tensors
from network import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score

run = Run.get_context()

def train( dataloader, learning_rate,batch_size,
          input_size,hidden_size, 
          nb_layers,dropout,
          val_dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_classes=2
    
    network = Network(device,batch_size, 
                      input_size,hidden_size,
                      nb_layers,dropout,
                      nb_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    

    
    # Train the model
    for epoch in range(nb_epochs):
        
        
        for i, (X, y) in enumerate(dataloader):
            X = X.reshape(-1, X.shape[1], input_size).to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            network.hidden_0 = network.init_hidden(batch_size = X.shape[0],
                                                   denominator = 1)
            network.hidden_1 = network.init_hidden(batch_size = X.shape[0],
                                                   denominator = 2)
            
            # Forward pass
            y_pred = network(X)
            loss = criterion(y_pred, y)

            # Backprop
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                run.log('loss', loss.item())
                
        # end of epoch       
        evaluate(val_dataloader, network, device)
        network.train()

    return network

def evaluate(dataloader, network, device):
    
    '''
        Evaluate model on validation set
        
        params:
            X_test: validation dataset
            y_test: validation target
            network: pytorch model
            device: torch device 
    '''
    with torch.no_grad(): 
         for i, (X, y) in enumerate(dataloader):
                network.hidden_0 = network.init_hidden(batch_size = X.shape[0],
                                                   denominator = 1)
                network.hidden_1 = network.init_hidden(batch_size = X.shape[0],
                                                       denominator = 2)
                
                X = X.reshape(-1, X.shape[1], X.shape[2]).to(device)
                y_pred = network(X)
                
                y_pred_np = y_pred.to('cpu').data.numpy()
                y_test_np = y.to('cpu').data.numpy()
                y_pred_np = np.argmax(y_pred_np, axis=1)

                precision = precision_score(y_test_np, y_pred_np)
                recall = recall_score(y_test_np, y_pred_np)
                f1 = f1_score(y_test_np, y_pred_np)

                run.log('precision', round(precision,2))
                run.log('recall', round(recall,2))
                run.log('f1', round(f1,2))


if __name__ == '__main__':
    
    print('Pytorch version', torch.__version__)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=.2, help='drop out')
    parser.add_argument('--layers', type=int,
                        default=1, help='number of layers')
    parser.add_argument('--hidden_units', type=int,
                        default=16, help='number of neurons')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Mini batch size')
    parser.add_argument('--data_path', type=str, 
                        help='path to training-set file')
    parser.add_argument('--output_dir', type=str, 
                        help='output directory')
    
    args = parser.parse_args()
    nb_epochs = args.epochs
    learning_rate = args.learning_rate
    dropout = args.dropout
    data_path = args.data_path
    output_dir = args.output_dir
    nb_layers = args.layers
    batch_size = args.batch_size
    
    hidden_size = args.hidden_units
    batch_size = args.batch_size
    
    print("Start training")
    
    print('learning rate', learning_rate)
    print('dropout', dropout)
    print('batch_size', batch_size)
    print('hidden_units', hidden_size)
    
    os.makedirs(data_path, exist_ok = True)
    training_file = os.path.join(data_path, 'preprocessed_train_file.csv')
    
    X, y = to_tensors(training_file)
    X_train, X_test, y_train, y_test = train_test_split(
                             X, y, test_size=0.15, random_state=122)
    input_size = X_train.shape[2]

    
    
    dataset = utils.TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train)) 
    dataloader = utils.DataLoader(dataset, batch_size = batch_size,
                                  shuffle = True)
    
    
    

    val_dataset = utils.TensorDataset(torch.from_numpy(X_test),
                                      torch.from_numpy(y_test))
    val_dataloader = utils.DataLoader(val_dataset, batch_size = batch_size,
                                      shuffle = True)
    
    
    network = train(dataloader,learning_rate,
                    batch_size,input_size,hidden_size, 
                    nb_layers,dropout,
                    val_dataloader)
    
    
    #evaluate(X_test,y_test, network, device)
    
    os.makedirs(output_dir, exist_ok = True)
    model_path = os.path.join(output_dir, 'network.pth')
    
    torch.save(network, model_path)
    run.register_model(model_name = 'network.pt', model_path = model_path)