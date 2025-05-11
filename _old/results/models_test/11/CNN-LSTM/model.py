import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_size, hid_size, n_layers, out_size, dropout_p, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        # bloco CNN simples (kernel_size=3, padding para manter T)
        self.conv = nn.Conv1d(
            in_channels=in_size,
            out_channels=in_size,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        # LSTM em vez de GRU
        self.lstm = nn.LSTM(
            in_size,
            hid_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        # camada final
        self.fc = nn.Linear(hid_size * self.num_directions, out_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (B, T, in_size)
        # CNN espera (B, C, T)
        x = x.transpose(1, 2)           # → (B, in_size, T)
        x = self.conv(x)               # → (B, in_size, T)
        x = self.relu(x)
        x = x.transpose(1, 2)          # → (B, T, in_size)

        # alimentar LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (n_layers * num_directions, B, hid_size)
        if self.num_directions == 2:
            fwd = h_n[-2]
            bwd = h_n[-1]
            h_cat = torch.cat((fwd, bwd), dim=1)
        else:
            h_cat = h_n[-1]

        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)
