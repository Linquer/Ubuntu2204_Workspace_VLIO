import torch.cuda
import numpy as np
import torch

class Config():
    def __init__(self):
        self.flow_list = [2, 4, 6, 8]
        self.state_dim = 8
        self.hidden_dim = 20
        self.num_layers = 4
        self.batch_size = 64
        self.num_epochs = 300
        self.learning_rate = 0.0001
        self.seed = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_config(self):
        print("flow_list: ", self.flow_list)
        print("state_dim: ", self.state_dim)
        print("batch_size: ", self.batch_size)
        print("hidden_dim: ", self.hidden_dim)
        print("num_layers: ", self.num_layers)
        print("num_epochs: ", self.num_epochs)
        print("learning_rate: ", self.learning_rate)
        print("seed: ", self.seed)
        print("device: ", self.device)
        



def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)