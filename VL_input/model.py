import torch
import torch.nn as nn
import torch.optim as optim

# 定义 Encoder 和 Decoder
class EncoderGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(EncoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden

class DecoderGRU(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(DecoderGRU, self).__init__()
        self.gru = nn.GRU(input_dim, output_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return outputs, hidden

'''
GRU Input: (batch_size, seq_len, input_dim)
output: (batch_size, seq_len, hidden_dim)
hidden: (num_layers, batch_size, hidden_dim)
'''

