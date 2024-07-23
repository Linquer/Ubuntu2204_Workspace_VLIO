import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_dataloader

def train_autoencoder(net, dataloaders, config):
    loss_list = []
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    net_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_dataloader_loss = 0
        for batch1, batch2, batch3, batch4, batch5 in zip(*dataloaders):
            epoch_dataloader_loss += batch_train_autoencoder(net, batch1, criterion, net_optimizer, config)
            epoch_dataloader_loss += batch_train_autoencoder(net, batch2, criterion, net_optimizer, config)
            epoch_dataloader_loss += batch_train_autoencoder(net, batch3, criterion, net_optimizer, config)
            epoch_dataloader_loss += batch_train_autoencoder(net, batch4, criterion, net_optimizer, config)
            epoch_dataloader_loss += batch_train_autoencoder(net, batch5, criterion, net_optimizer, config)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_dataloader_loss/(len(dataloaders[0])*len(dataloaders)):.4f}")
        loss_list.append(epoch_dataloader_loss/(len(dataloaders[0])*len(dataloaders)))
    return loss_list
            

def batch_train_autoencoder(net, batch, criterion, net_optimizer, config):
    batch = batch.to(config.device)
    net.zero_grad()
    decoder_outputs, _ = net(batch)
    loss = criterion(decoder_outputs, batch.reshape(batch.shape[0], -1))
    loss.backward()
    net_optimizer.step() 
    return loss.item()


def test_model(net, config, loacl_model=False, model_path=None):
    if loacl_model:
        net = torch.load(model_path)
    dataloaders = create_dataloader(config.flow_list, config.batch_size, config.state_dim)
    for data_loader in dataloaders:
        for batch_data in data_loader:
            batch_data = batch_data.to(config.device)
            decoder_outputs, encoder_hidden = net(batch_data)
            print(batch_data[10])
            print(decoder_outputs[10])
            print(batch_data[20])
            print(decoder_outputs[20])
            break