

import torch 
import torch.nn as nn
import torch.utils.data as utils
from network import Network
from sklearn.metrics import precision_score,recall_score,f1_score



def train(X_train,y_train, 
          X_val,y_val, 
          learning_rate,batch_size,
          hidden_size, nb_layers,
          dropout, run):
    
    dataset = utils.TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train)) 
    dataloader = utils.DataLoader(dataset, batch_size = batch_size,
                                  shuffle = True)
    
    val_dataset = utils.TensorDataset(torch.from_numpy(X_val),
                                      torch.from_numpy(y_val))
    val_dataloader = utils.DataLoader(val_dataset)
    
    use_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_size = X_train.shape[2]
    network = Network(batch_size, 
                      input_size,hidden_size,
                      nb_layers,dropout).to(use_gpu)
    
    # Loss and optimizer
    cost_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(nb_epochs):
        
        
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            X = X.to(use_gpu)
            y = y.to(use_gpu)
            y_pred = network(X)
            loss = cost_fn(y_pred, y)

            # Backprop
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                run.log('loss', loss.item())
                
        # end of epoch       
        evaluate(val_dataloader, network, use_gpu)
        network.train()

    return network

def evaluate(dataloader, network, use_gpu, run):
    
    '''
        Evaluate model on validation set
        
        params:
            dataloader: dataloader
            network: model
            use_gpu: device
            run: AML RUN
    '''
    
    
    
    y_pred_lst = []
    y_truth_lst = []
    with torch.no_grad(): 
         
        for i, (X, y) in enumerate(dataloader):
            
                X = X.to(use_gpu)
                output = network(X)
                
                print("SIZE",output.size())
                y_pred = output.max(1, keepdim=True)[1]
                print(y_pred)
                
                y_pred_lst.append(y_pred)
                y_truth_lst.append(y.data.numpy().reshape(-1))
                
                '''y_pred_np = y_pred.to('cpu').data.numpy()
                y_test_np = y.to('cpu').data.numpy()
                y_pred_np = np.argmax(y_pred_np, axis=1)'''
                
        y_pred_np = np.array(y_pred_lst)
        y_truth_np = np.array(y_truth_lst)
        
        precision = precision_score(y_truth_np, y_pred_np)
        recall = recall_score(y_truth_np, y_pred_np)
        f1 = f1_score(y_truth_np, y_pred_np)

        run.log('precision', round(precision,2))
        run.log('recall', round(recall,2))
        run.log('f1', round(f1,2))