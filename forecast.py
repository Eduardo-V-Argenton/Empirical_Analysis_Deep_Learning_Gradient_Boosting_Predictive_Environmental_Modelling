# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from datetime import datetime, timedelta
from plotnine import ggplot, aes, geom_line, labs, theme_bw, facet_wrap
from basic import Model, features_y, features_X

# Configurações
MODEL_PATH = 'results/main/9/model.h5'
X_SCALER_PATH = 'results/main/9/x_scaler.pkl'
Y_SCALER_PATH = 'results/main/9/y_scaler.pkl'
DATA_PATH = 'data/cleaned_data.csv'
WINDOW_SIZE = 48
STEPS = 24

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(16, 256, 2, 8, 0.4, True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

x_scaler = load(X_SCALER_PATH)
y_scaler = load(Y_SCALER_PATH)

df = pd.read_csv(DATA_PATH)
raw_window = df[features_X].iloc[-WINDOW_SIZE:]
window_norm = x_scaler.transform(raw_window)
current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

# %%
predictions = []

# Suponha que os dados terminam em "agora"
# Crie uma hora inicial artificial para simular o tempo
now = datetime.now().replace(minute=0, second=0, microsecond=0)

# %%
for step in range(STEPS):
    with torch.inference_mode():
        pred_norm = model(current).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_norm)[0]

    # Simular timestamp apenas para calcular as features temporais
    sim_time = now + timedelta(hours=step + 1)
    hour = sim_time.hour
    doy  = sim_time.timetuple().tm_yday

    # Geração das múltiplas harmônicas de hora e do dia do ano
    row = {}
    for col in features_X:
        if col in features_y:
            row[col] = pred[features_y.index(col)]
        elif col.startswith('hour_sin'):
            harm = int(col[-1])
            row[col] = np.sin(2 * np.pi * harm * hour / 24)
        elif col.startswith('hour_cos'):
            harm = int(col[-1])
            row[col] = np.cos(2 * np.pi * harm * hour / 24)
        elif col == 'doy_sin':
            row[col] = np.sin(2 * np.pi * doy / 365)
        elif col == 'doy_cos':
            row[col] = np.cos(2 * np.pi * doy / 365)
        else:
            # exógenas fixas (ex: zero ou médias)
            row[col] = 0

    # Normalizar, atualizar janela e fazer próxima previsão
    df_row = pd.DataFrame([row], columns=features_X)
    norm_row = x_scaler.transform(df_row)
    window_norm = np.vstack([window_norm[1:], norm_row])
    current = torch.tensor(window_norm).float().unsqueeze(0).to(device)

    predictions.append({'Step': step + 1, **row})

# %%
# Resultados
forecast_df = pd.DataFrame(predictions)
print(forecast_df[['Step'] + features_y])

# Salvar CSV
forecast_df.to_csv(f'previsao_proximas_{STEPS}_h.csv', index=False)

# Plot
from plotnine import scale_x_continuous
plot_df = forecast_df.melt(id_vars='Step', value_vars=features_y,
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
