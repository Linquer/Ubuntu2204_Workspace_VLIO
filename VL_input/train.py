import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_dataloader

def train_autoencoder(net, dataloaders, config):
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    net_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_dataloader_loss = 0
        for dataloader in dataloaders:
            epoch_dataloader_loss = 0
            for batch in dataloader:
                batch = batch.to(config.device)
                net.zero_grad()
                decoder_outputs, encoder_hidden = net(batch)
                # 计算损失并反向传播
                loss = criterion(decoder_outputs, batch.reshape(batch.shape[0], -1))
                loss.backward()
                net_optimizer.step()
                epoch_dataloader_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_dataloader_loss/(len(dataloader)):.4f}")

        
def test_model(net, config):
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