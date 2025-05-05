# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from datetime import datetime, timedelta
from plotnine import ggplot, aes, geom_line, labs, theme_bw, facet_wrap, scale_x_continuous
from basic import Model,features_y,features_X

# Configurações
MODEL_PATH = 'results/main/10/model.h5'
X_SCALER_PATH = 'results/main/10/x_scaler.pkl'
Y_SCALER_PATH = 'results/main/10/y_scaler.pkl'
DATA_PATH = 'data/cleaned_data.csv'
WINDOW_SIZE = 12
STEPS = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Configuração do dispositivo e carregamento do modelo
model = Model(18, 64, 1, 8, 0.1, True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# %%
model.eval()
# Carregamento dos scalers e dados
x_scaler = load(X_SCALER_PATH)
y_scaler = load(Y_SCALER_PATH)
df = pd.read_csv(DATA_PATH)

# Pegar a última janela de dados
raw_window = df[features_X].iloc[-WINDOW_SIZE:]
window_norm = x_scaler.transform(raw_window)
current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

# %%
predictions = []

# Obter o último timestamp do dataframe para continuar a partir dele
last_timestamp = pd.to_datetime(df["Timestamp"].iloc[-1])
# Usar o último timestamp como referência, caso contrário usar agora
now = last_timestamp if not pd.isnull(last_timestamp) else datetime.now().replace(minute=0, second=0, microsecond=0)

# Último delta_t conhecido (intervalo de tempo em segundos entre registros)
try:
    last_delta_t = df["delta_t"].iloc[-1]
    # Se não houver valor, assumir 1 hora (3600 segundos)
    if pd.isnull(last_delta_t):
        last_delta_t = 3600
except (KeyError, IndexError):
    last_delta_t = 3600  # 1 hora em segundos

# %%
for step in range(STEPS):
    with torch.inference_mode():
        pred_norm = model(current).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_norm)[0]

    # Simular o próximo timestamp
    sim_time = now + timedelta(hours=step + 1)
    hour = sim_time.hour
    doy = sim_time.timetuple().tm_yday
    dow = sim_time.weekday()  # 0 é segunda-feira, 6 é domingo

    # Calcular as features temporais
    ts_unix = sim_time.timestamp()

    # Para ts_norm, precisaríamos conhecer o método original de normalização
    # Vamos assumir que é baseado no unix timestamp dividido por algum fator de escala
    # Usaremos a última proporção conhecida entre ts_unix e ts_norm
    try:
        ts_norm_ratio = df["ts_norm"].iloc[-1] / df["ts_unix"].iloc[-1]
        ts_norm = ts_unix * ts_norm_ratio
    except (KeyError, IndexError, ZeroDivisionError):
        ts_norm = ts_unix / 86400  # normalização simples (segundos em um dia)

    # Preparar row com as features previstas
    row = {}

    # Preencher com valores previstos
    for i, feature in enumerate(features_y):
        row[feature] = pred[i]

    # Adicionar features temporais
    row["ts_unix"] = ts_unix
    row["ts_norm"] = ts_norm
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # delta_t é a diferença de tempo em segundos
    row["delta_t"] = last_delta_t  # Mantemos o mesmo delta_t por simplicidade

    # Para delta_t_norm, seguimos a mesma lógica de ts_norm
    try:
        delta_t_norm_ratio = df["delta_t_norm"].iloc[-1] / df["delta_t"].iloc[-1]
        row["delta_t_norm"] = row["delta_t"] * delta_t_norm_ratio
    except (KeyError, IndexError, ZeroDivisionError):
        row["delta_t_norm"] = row["delta_t"] / 86400  # normalização simples

    # Criar DataFrame com todas as features necessárias
    missing_features = set(features_X) - set(row.keys())
    for feature in missing_features:
        # Para features não previstas nem calculadas, usamos o último valor conhecido
        try:
            row[feature] = df[feature].iloc[-1]
        except (KeyError, IndexError):
            row[feature] = 0  # Valor padrão se não encontrado

    # Normalizar a nova linha e atualizar a janela
    df_row = pd.DataFrame([row], columns=features_X)
    norm_row = x_scaler.transform(df_row)
    window_norm = np.vstack([window_norm[1:], norm_row])
    current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

    # Adicionar timestamp às previsões para melhor visualização
    forecast_timestamp = sim_time.strftime('%Y-%m-%d %H:%M:%S')
    predictions.append({'Step': step + 1, 'Timestamp': forecast_timestamp, **{f: row[f] for f in features_y}})

# %%
# Criar DataFrame com as previsões
forecast_df = pd.DataFrame(predictions)

# Exibir resultados
print(forecast_df[['Step', 'Timestamp'] + features_y])

# Salvar CSV
forecast_df.to_csv(f'previsao_proximas_{STEPS}_h.csv', index=False)

# Plot
plot_df = forecast_df.melt(id_vars=['Step', 'Timestamp'], value_vars=features_y,
                         var_name='Variável', value_name='Valor')

plot = (
    ggplot(plot_df, aes(x='Step', y='Valor', color='Variável'))
    + geom_line(size=1)
    + facet_wrap('~Variável', scales='free_y', ncol=2)
    + labs(title=f'Previsão para as Próximas {STEPS} Horas',
         x='Horas à Frente', y='Valor Previsto')
    + theme_bw()
    + scale_x_continuous(breaks=range(0, STEPS + 1, 6))
)

plot.save('previsao.png', dpi=300, height=8, width=10)
print(plot)
# %%
print(plot_df)
