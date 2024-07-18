from model import DecoderGRU, EncoderGRU
import torch
import torch.nn as nn

state_dim = 8
hidden_dim = 40
state_2 = torch.randn(10, state_dim)
state_4 = torch.randn(10, state_dim*2)
state_6 = torch.randn(10, state_dim*3)
state_8 = torch.randn(10, state_dim*4)

state_2 = state_2.view(10, -1, state_dim)
state_4 = state_4.view(10, -1, state_dim)
state_6 = state_6.view(10, -1, state_dim)
state_8 = state_8.view(10, -1, state_dim)

encoder_net = EncoderGRU(state_dim, hidden_dim)
decoder_net = DecoderGRU(hidden_dim, state_dim)

for data in [state_2, state_4, state_6, state_8]:
    en_output, en_hidden = encoder_net(data)
    en_hidden = en_hidden.permute(1, 0, 2).repeat(1, data.shape[1], 1)
    dec_output, dec_hidden = decoder_net(en_output)
    print(dec_output.shape)

# encoder_net = EncoderGRU(4, 20)
# output, hidden = encoder_net(data)

# hidden_new = hidden.permute(1, 0, 2).repeat(1, N, 1)

