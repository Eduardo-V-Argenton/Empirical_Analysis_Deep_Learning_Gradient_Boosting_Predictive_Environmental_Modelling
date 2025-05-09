import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_size, hid_size, n_layers, out_size, dropout_p, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            in_size,
            hid_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0,  # Dropout só entre camadas se n_layers > 1
            bidirectional=bidirectional
        )
        # Camada final
        self.fc = nn.Linear(hid_size * self.num_directions, out_size)
        # Dropout externo (opcional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (B, T, in_size)
        gru_out, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, B, hid_size)
        if self.num_directions == 2:
            # Última camada forward + backward
            fwd = h_n[-2]  # última camada, direção forward
            bwd = h_n[-1]  # última camada, direção backward
            h_cat = torch.cat((fwd, bwd), dim=1)
        else:
            # Só pegar o último hidden state da última camada
            h_cat = h_n[-1]

        # Dropout antes da FC (se desejado)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


features_y = [
    'Temperature', 'Precipitation_log', 'Humidity', 'Wind_Speed_kmh',
    'Soil_Moisture', 'Soil_Temperature',
    'Wind_Dir_Sin', 'Wind_Dir_Cos'
]

features_X = [
    'Temperature','Humidity','Wind_Speed_kmh','Soil_Moisture',
    'Soil_Temperature','Wind_Dir_Sin','Wind_Dir_Cos','Precipitation_log',
    'ts_unix','ts_norm','hour_sin','hour_cos','doy_sin','doy_cos','dow_sin',
    'dow_cos','delta_t','delta_t_norm'
]
