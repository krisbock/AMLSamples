

import torch 
import torch.nn as nn
import torch.utils.data as utils
from azureml.core import Run
import numpy as np
import pandas as pd
from utils import tensorize,to_tensors
from network import Network
from sklearn.metrics import (recall_score, 
                             precision_score, 
                             accuracy_score)

run = Run.get_context()

def train( dataloader, learning_rate,
          device,input_size, 
          hidden_size, nb_layers,
          dropout, nb_classes,
         X_val,y_val):
    
    
    
    network = Network(device, input_size,
                      hidden_size, nb_layers, dropout, 
                      nb_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    

    
    # Train the model
    for epoch in range(nb_epochs):
        for i, (X, y) in enumerate(dataloader):
            X = X.reshape(-1, X.shape[1], input_size).to(device)
            y = y.to(device)

            # Forward pass
            y_pred = network(X)
            loss = criterion(y_pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                '''print('epoch [{}/{}], loss: {:.4f}'
                                   .format(epoch+1,nb_epochs, loss.item()))'''
                run.log('loss', loss.item())
                
        evaluate(X_val,y_val, network, device)

    return network

def evaluate(X_test,y_test , network, device):
    
    '''
        Evaluate model on testing set
        
        params:
            testfile_path: path to testing file
    '''

    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    y_pred = network(X_test)
    
    y_pred_np = y_pred.to('cpu').data.numpy()
    y_test_np = y_test.to('cpu').data.numpy()
    y_pred_np = np.argmax(y_pred_np, axis=1)
    
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(y_test_np, y_pred_np)
    recall = recall_score(y_test_np, y_pred_np)
    
    run.log('accuracy', accuracy)
    run.log('precision', precision)
    run.log('recall', recall)

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
    nb_classes = 2
    batch_size = args.batch_size
    
    print("Start training")
    
    run.log('learning rate', learning_rate)
    run.log('dropout', dropout)
    run.log('batch_size', batch_size)
    run.log('hidden_units', hidden_size)
    
    os.makedirs(data_path, exist_ok = True)
    training_file = os.path.join(data_path, 'preprocessed_train_file.csv')
    
    X_train, y_train = to_tensors(training_file)
    input_size = X_train.shape[2]
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    
    val_file_path = os.path.join(data_path, 'preprocessed_test_file.csv')
    X_val,y_val = to_tensors(val_file_path, istest = True)
    
    dataset = utils.TensorDataset(X_train,y_train) 
    dataloader = utils.DataLoader(dataset, batch_size = batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    network = train(dataloader,learning_rate,
                    device, input_size,
                    hidden_size, nb_layers,
                    dropout, nb_classes,
                    X_val,y_val)
    
    
    evaluate(X_val,y_val, network, device)
    
    os.makedirs(output_dir, exist_ok = True)
    model_path = os.path.join(output_dir, 'network.pth')
    torch.save(network, model_path)
    run.register_model(model_name = 'network.pth', model_path = model_path)