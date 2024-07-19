model_v1: 最原始的版本，只有一层GRU。Encoder 输出的 hidden，需要复制 N 次，作为 Decoder 的输入。

model_v2: 采用 4 层 GRU，Encoder 的输出 hidden 一共有四个，一起输入给 Decoder。此时 decoder 的输出包括(batch, 4, state_dim)。对于就一个 state_dim 的状态，就只采纳第一个 state_dim。decoder_outputs = decoder_outputs[:, 0:x_shape[1], :]

model_v3: 采用 4 层GRU，Encoder 的输出 hidden 一共有四个，但是只拿最后一层输出的 hidden，由于就一个因此需要加大 hidden 输出的维度。并且根据输入数据 seq 的长度，对 hidden 复制 N 次。然后再作为 decoder 的输入。