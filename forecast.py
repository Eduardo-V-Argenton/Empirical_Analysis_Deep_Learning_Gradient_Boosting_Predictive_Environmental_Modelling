import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from datetime import datetime, timedelta
from plotnine import ggplot, aes, geom_line, labs, theme_bw, scale_x_datetime, facet_wrap

# Configuração inicial
MODEL_PATH = 'results/main/8/model.h5'  # Substitua pelo seu caminho
X_SCALER_PATH = 'results/main/8/x_scaler.pkl'  # Substitua
Y_SCALER_PATH = 'results/main/8/y_scaler.pkl'  # Substitua
DATA_PATH = 'data/cleaned_data.csv'
WINDOW_SIZE = 6
FEATURES_Y = ['Temperature', 'Precipitation_log', 'Humidity', 'Wind_Speed_kmh',
              'Soil_Moisture', 'Soil_Temperature', 'Wind_Dir_Sin', 'Wind_Dir_Cos']

# 1. Carregar recursos necessários
class GRU(nn.Module):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRU(10, 128, 4, 8, 0.4, True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_scaler = load(X_SCALER_PATH)
y_scaler = load(Y_SCALER_PATH)

# 2. Preparar dados iniciais
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Pegar última janela de dados

features_X = df.columns.drop(['Timestamp', 'Wind_Direction', 'Precipitation'])
raw_window = df.drop(columns=['Timestamp', 'Wind_Direction', 'Precipitation']).iloc[-WINDOW_SIZE:]
from datetime import timedelta

# pega a última janela já normalizada
window_norm = x_scaler.transform(raw_window)  # shape (6, n_features)
current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

predictions = []
last_ts = df['Timestamp'].iloc[-1]

def hour_sin_cos(ts):
    rad = 2 * np.pi * ts.hour / 24
    return np.sin(rad), np.cos(rad)

steps = 48
for _ in range(steps):
    with torch.no_grad():
        pred_norm = model(current).cpu().numpy()  # (1, n_targets)
    pred = y_scaler.inverse_transform(pred_norm)[0]

    # cria novo timestamp e features temporais
    new_ts = last_ts + timedelta(hours=1)
    hs, hc   = hour_sin_cos(new_ts)
    doy_rad  = 2*np.pi * new_ts.timetuple().tm_yday / 365
    dys, dyc = np.sin(doy_rad), np.cos(doy_rad)

    # monta a linha completa na ordem features_X
    row = {}
    for col in features_X:
        if col in FEATURES_Y:
            row[col] = pred[FEATURES_Y.index(col)]
        elif col == 'hour_sin':
            row[col] = hs
        elif col == 'hour_cos':
            row[col] = hc
        elif col == 'dayofyear_sin':
            row[col] = dys
        elif col == 'dayofyear_cos':
            row[col] = dyc
        else:
            # se houver mais exógenas, trate aqui
            row[col] = 0

    # normalize e atualiza janela
    df_row = pd.DataFrame([row], columns=features_X)
    norm_row = x_scaler.transform(df_row)
    window_norm = np.vstack([window_norm[1:], norm_row])
    current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

    predictions.append({'Timestamp': new_ts, **row})
    last_ts = new_ts


# 5. Resultado final
forecast_df = pd.DataFrame(predictions)
print("\nPrevisão para as próximas 12 horas:")
print(forecast_df[['Timestamp'] + FEATURES_Y])

# Opcional: Salvar em CSV
forecast_df.to_csv(f'previsao_proximas_{steps/2}_h.csv', index=False)
# 6. Plotar resultados
# Preparar dados para plotagem
plot_df = forecast_df.melt(id_vars='Timestamp',
                          value_vars=FEATURES_Y,
                          var_name='Variável',
                          value_name='Valor')

# Criar gráfico
plot = (
    ggplot(plot_df, aes(x='Timestamp', y='Valor', color='Variável'))
    + geom_line(size=1)
    + facet_wrap('~Variável', scales='free_y', ncol=2)
    + labs(title=f'Previsão das Próximas {steps/2} Horas',
          x='Horário',
          y='Valor Previsto')
    + theme_bw()
    + scale_x_datetime(date_labels='%H:%M')
)

# Salvar e mostrar
plot.save('previsao.png', dpi=300, height=8, width=10)
print(plot)
