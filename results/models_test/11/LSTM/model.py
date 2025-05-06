import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_size, hid_size, n_layers, out_size, dropout_p, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hid_size * self.num_directions, out_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (B, T, in_size)
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hid_size)
        if self.num_directions == 2:
            # Última camada forward e backward
            fwd = h_n[-2]
            bwd = h_n[-1]
            h_cat = torch.cat((fwd, bwd), dim=1)
        else:
            h_cat = h_n[-1]
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)
