# from model import AutoencoderGRU
from model_v4 import AutoencoderGRU
from data_loader import create_dataloader
from config import Config, set_seed
from train import train_autoencoder, test_model
import torch

config = Config()
set_seed()
config.print_config()

if __name__ == '__main__':
    train_dataloader = \
            create_dataloader(config.flow_list, config.batch_size, config.state_dim)
    net = AutoencoderGRU(config.state_dim, config.hidden_dim, config.state_dim, config.num_layers).to(config.device)

    train_autoencoder(net, train_dataloader, config)

    test_model(net, config)

    
