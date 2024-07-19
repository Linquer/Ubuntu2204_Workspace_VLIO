import torch
import torch.nn as nn

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
        return outputs, hidden

class AutoencoderGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(AutoencoderGRU, self).__init__()
        self.encoder = EncoderGRU(input_dim, hidden_dim, num_layers=num_layers)
        self.decoder = DecoderGRU(hidden_dim, output_dim, num_layers=num_layers)
        self.deal_encoder_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.deal_decoder_hidden = nn.Sequential(
            nn.Linear(output_dim, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim)
        )   
        
    
    def forward(self, x):
        x_shape = x.shape
        encoder_output, encoder_hidden = self.encoder(x)
        encoder_hidden = encoder_hidden[-1]
        encoder_hidden = encoder_hidden.unsqueeze(dim=1)
        encoder_hidden = encoder_hidden.repeat(1, x_shape[1], 1)
        encoder_hidden = 0.7 * encoder_hidden + 0.3 * encoder_output[:, 0: x_shape[1], :]
        
        decoder_outputs, _ = self.decoder(encoder_hidden)
        decoder_outputs = self.deal_decoder_hidden(decoder_outputs)
        decoder_outputs = decoder_outputs.reshape(decoder_outputs.shape[0], -1)
        return decoder_outputs, encoder_hidden

'''
GRU Input: (batch_size, seq_len, input_dim)
output: (batch_size, seq_len, hidden_dim)
hidden: (num_layers, batch_size, hidden_dim)
'''

