import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from datetime import datetime, timedelta
from plotnine import ggplot, aes, geom_line, labs, theme_bw, scale_x_datetime, facet_wrap

# Configuração inicial
MODEL_PATH = 'results/main/7/model.h5'  # Substitua pelo seu caminho
X_SCALER_PATH = 'results/main/7/x_scaler.pkl'  # Substitua
Y_SCALER_PATH = 'results/main/7/y_scaler.pkl'  # Substitua
DATA_PATH = 'data/cleaned_data.csv'
WINDOW_SIZE = 12
FEATURES_Y = ['Temperature', 'Precipitation', 'Humidity', 'Wind_Speed_kmh',
              'Soil_Moisture', 'Soil_Temperature', 'Wind_Dir_Sin', 'Wind_Dir_Cos']

# 1. Carregar recursos necessários
class LSTM(torch.nn.Module):
    def __init__(self, in_size=10, hid_size=256, num_layers=1, out_size=8, dropout=0.27):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            in_size, hid_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = torch.nn.Linear(hid_size, out_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_scaler = load(X_SCALER_PATH)
y_scaler = load(Y_SCALER_PATH)

# 2. Preparar dados iniciais
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Pegar última janela de dados
raw_window = df.drop(columns=['Timestamp', 'Wind_Direction']).iloc[-WINDOW_SIZE:]
window_normalized = x_scaler.transform(raw_window)

# 3. Função auxiliar para gerar features temporais
def generate_time_features(timestamp):
    return {
        'Hour': timestamp.hour,
        'Day': timestamp.day,
        'Month': timestamp.month
    }

# 4. Fazer previsões
current_input = torch.tensor(window_normalized).float().unsqueeze(0).to(device)
predictions = []
last_timestamp = df['Timestamp'].iloc[-1]

for _ in range(240):
    with torch.no_grad():
        pred_normalized = model(current_input).cpu().numpy()

    # Desnormalizar apenas as features Y
    pred = y_scaler.inverse_transform(pred_normalized)[0]

    # Criar novo timestamp
    new_timestamp = last_timestamp + timedelta(hours=1)

    # Gerar novas features exógenas
    time_features = generate_time_features(new_timestamp)

    # Criar nova linha de dados
    new_row = {}
    for col in raw_window.columns:
        if col in FEATURES_Y:
            new_row[col] = pred[FEATURES_Y.index(col)]
        else:
            new_row[col] = time_features.get(col.split('_')[0], 0)  # Ajuste conforme suas features

    # Atualizar janela deslizante
    window_normalized = np.vstack([window_normalized[1:], x_scaler.transform([list(new_row.values())])])
    current_input = torch.tensor(window_normalized).float().unsqueeze(0).to(device)

    predictions.append({
        'Timestamp': new_timestamp,
        **new_row
    })
    last_timestamp = new_timestamp

# 5. Resultado final
forecast_df = pd.DataFrame(predictions)
print("\nPrevisão para as próximas 12 horas:")
print(forecast_df[['Timestamp'] + FEATURES_Y])

# Opcional: Salvar em CSV
forecast_df.to_csv('previsao_proximas_12h.csv', index=False)
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
    + labs(title='Previsão das Próximas 12 Horas',
          x='Horário',
          y='Valor Previsto')
    + theme_bw()
    + scale_x_datetime(date_labels='%H:%M')
)

# Salvar e mostrar
plot.save('previsao.png', dpi=300, height=8, width=10)
print(plot)
