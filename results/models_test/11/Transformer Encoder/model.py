import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_size, hid_size, n_layers, out_size, dropout_p, n_heads):
        super().__init__()
        self.input_proj = nn.Linear(in_size, hid_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_size, nhead=n_heads, dropout=dropout_p, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(hid_size, out_size)

    def forward(self, x):
        # x: (B, T, in_size)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])  # pega a última posição temporal
