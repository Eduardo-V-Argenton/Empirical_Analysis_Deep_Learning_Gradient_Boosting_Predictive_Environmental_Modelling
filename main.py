# %%
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, vonmises, gamma,lognorm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme,facet_wrap,scale_color_manual,theme_bw
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from pandas.api.types import CategoricalDtype
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from typing import List, Any

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
print(torch.__version__)  # Verifique a versão
print(torch.cuda.is_available())  # Deve retornar True
print(torch.version.hip)  # Verifique a versão HIP
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
# %%
df = pd.read_csv('data/kelowna_weather_2024.csv')
df.describe()

# %%
# df = pd.read_csv('kelowna_weather_2024.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df = df.resample('h').mean().interpolate(method='time', limit_direction='both')
# %%
df.describe()

# %%
decomps = {}
residuals = pd.DataFrame(index=df.index)
for col in df.columns:
    if col == 'Timestamp':
        continue
    stl_res = STL(df[col], period=24, robust=True).fit()
    decomps[col] = stl_res
    residuals[col] = stl_res.resid

# %%
iso = IsolationForest(contamination='auto', random_state=42)
iso.fit(residuals)

# %%
flags = iso.predict(residuals)
is_multivariate_anomaly = pd.Series(flags == -1, index=residuals.index)
df['is_multivariate_anomaly'] = is_multivariate_anomaly
df.head()

# %%
print('Foram encontrados {} outliers'.format(is_multivariate_anomaly.sum()))

# %%
df_clean = df.copy()
for col, stl_res in decomps.items():
    cleaned_series = df[col].copy()
    cleaned_series[is_multivariate_anomaly] = (stl_res.trend + stl_res.seasonal)[is_multivariate_anomaly]
    df_clean[col] = cleaned_series
df_clean.drop(columns=['is_multivariate_anomaly'], inplace=True)
df.drop(columns=['is_multivariate_anomaly'], inplace=True)
df_clean.head()

# %%
df_clean['Temperature'] = df_clean['Temperature'].round(3)
df_clean['Precipitation'] = df_clean['Precipitation'].round(3)
df_clean['Humidity'] = df_clean['Humidity'].round(3)
df_clean['Wind_Speed_kmh'] = df_clean['Wind_Speed_kmh'].round(3)
df_clean['Wind_Direction'] = df_clean['Wind_Direction'].round(3)
df_clean['Soil_Moisture'] = df_clean['Soil_Moisture'].round(3)
df_clean['Soil_Temperature'] = df_clean['Soil_Temperature'].round(3)
df_clean.head()
# %%
print('--- Original ---')
print(df.describe())
print('--- Limpo ---')
print(df_clean.describe())

# %%
orig_long = (
    df
    .reset_index()
    .melt(id_vars='Timestamp',
          var_name='variable',
          value_name='value')
    .assign(type='original')
)

clean_long = (
    df_clean
    .reset_index()
    .melt(id_vars='Timestamp',
          var_name='variable',
          value_name='value')
    .assign(type='cleaned')
)

combined = pd.concat([orig_long, clean_long], ignore_index=True)
type_order = CategoricalDtype(['original', 'cleaned'], ordered=True)
combined['type'] = combined['type'].astype(type_order)

plot = (
    ggplot(combined, aes('Timestamp', 'value', color='type'))
    + geom_line()
    + facet_wrap('~variable', scales='free_y', ncol=2)
    + scale_color_manual(values=['firebrick', 'navy'])
    + labs(
        title='Comparação: valores originais vs pós-tratamento',
        x='Timestamp',
        y='Valor',
        color='Série'
    )
    + theme(figure_size=(10, 6))
)

plot.show()
# %%
df_clean.reset_index(inplace=True)
df = df_clean

# %%
# profile = ProfileReport(df, title='Profile Report')
# profile.to_file('profile_report.html')

# %%
# 1. Transformar o Timestamp em variáveis úteis
df['hour'] = df['Timestamp'].dt.hour

# 2. Transformar Wind_Direction (porque ângulo 0° e 360° são "iguais")
df['Wind_Dir_Sin'] = np.sin(np.deg2rad(df['Wind_Direction']))
df['Wind_Dir_Cos'] = np.cos(np.deg2rad(df['Wind_Direction']))

# %%
# Features que serão usadas
features_thermo_hydro = ['Temperature', 'Humidity', 'Soil_Temperature', 'Soil_Moisture']
features_precipitation = ['Precipitation']
features_wind = ['Wind_Speed_kmh', 'Wind_Dir_Sin', 'Wind_Dir_Cos']

common_features = features_thermo_hydro + features_precipitation + features_wind + ['hour']
# %%
class LSTM(nn.Module):
    def __init__(self, in_size, hid_size, num_layers, out_size):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hid_size, num_layers, batch_first=True,dropout=0.2)
        self.fc   = nn.Linear(hid_size, out_size)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)           # h_n: (num_layers, B, hid_size)
        h_last = h_n[-1]                     # pega a saída da última camada
        return self.fc(h_last)              # (B, out_size)

# %%
X = df[common_features]
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

X_scaler = MinMaxScaler((0,1)); X_scaler.fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

def create_sequences(Xs, ys, window_size):
    X_seq, y_seq = [], []
    for i in range(len(Xs) - window_size):
        X_seq.append(Xs[i:i+window_size])
        y_seq.append(ys[i+window_size])
    return np.array(X_seq), np.array(y_seq)

# %%
class ModelConfig:
    __slots__ = (
        "name",
        "y_features", "y",
        "train_data", "test_data",
        "scaler",
        "train_scaled", "test_scaled",
        "X_train", "y_train", "X_test", "y_test",
        "input_size", "output_size",
        "hidden_size", "num_layers", "lr", "batch_size", "epochs",
        "train_loader", "model", "optimizer", "criterion",
        "scheduler", "window_size", "train_losses", "metrics_per_feature"
    )

    def __init__(self, name: str, y_features: List[str], y: Any):
        self.name = name
        self.y_features = y_features
        self.y = y
        # demais atributos serão atribuídos abaixo

# então crie instâncias em vez de dicts
thermo_hydro = ModelConfig("thermo_hydro", features_thermo_hydro, df[features_thermo_hydro])
precip = ModelConfig("precipitation", features_precipitation, df[features_precipitation])
wind   = ModelConfig("wind",        features_wind,        df[features_wind])

for cfg in (thermo_hydro, precip, wind):
    # atribua train/test raw
    cfg.train_data = cfg.y[:train_size]
    cfg.test_data  = cfg.y[train_size:]
    # scaler para y
    cfg.scaler     = MinMaxScaler((0,1)).fit(cfg.train_data)
    # escalonados
    cfg.train_scaled = cfg.scaler.transform(cfg.train_data)
    cfg.test_scaled  = cfg.scaler.transform(cfg.test_data)
    # sequências
    cfg.window_size = 25
    cfg.X_train, cfg.y_train = create_sequences(X_train_scaled, cfg.train_scaled, cfg.window_size)
    cfg.X_test,  cfg.y_test  = create_sequences(X_test_scaled,  cfg.test_scaled,  cfg.window_size)
    # parâmetros do modelo
    cfg.input_size  = cfg.X_train.shape[2]
    cfg.output_size = cfg.y_train.shape[1]
    cfg.hidden_size = 32
    cfg.num_layers  = 1
    cfg.lr          = 1e-3
    cfg.batch_size  = 32
    cfg.epochs      = 100
    # DataLoader
    cfg.train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(cfg.X_train).float(),
            torch.from_numpy(cfg.y_train).float()
        ),
        cfg.batch_size, shuffle=True
    )
    cfg.model = LSTM(cfg.input_size, cfg.hidden_size, cfg.num_layers, cfg.output_size).to(device)
    cfg.optimizer = torch.optim.Adam(cfg.model.parameters(), lr=cfg.lr)
    cfg.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cfg.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    cfg.criterion = nn.MSELoss()
    cfg.train_losses = []
    cfg.metrics_per_feature = []

print(thermo_hydro, precip, wind)
# %%
# Alterações

# %%
# Loop de treino
for cfg in (thermo_hydro, precip, wind):
    print(f"-----{cfg.name}-----")
    best_avg_loss = float('inf')
    epochs_since_best = 0
    for epoch in range(1, cfg.epochs+1):
        cfg.model.train()
        total_loss = 0
        for xb, yb in cfg.train_loader:
            cfg.optimizer.zero_grad()
            pred = cfg.model(xb.to(device))            # (B, n_features)
            loss = cfg.criterion(pred, yb.to(device))
            loss.backward()
            cfg.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(cfg.train_loader.dataset)
        cfg.train_losses.append(avg_loss)
        cfg.scheduler.step(avg_loss)

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            epochs_since_best = 0
            # print(f'New Best Avg Loss: {best_avg_loss:.5f}')
        else:
            epochs_since_best += 1

            # Early stopping baseado na média móvel
            if epochs_since_best > 20:
                print(f'Early stopping at epoch {epoch}')
                break
        if(epoch % 10 == 0):
            print(f"Epoch {epoch:02d}, Loss {avg_loss:.4f}")

# %%
all_y_pred = []
all_y_true = []
for cfg in (thermo_hydro, precip, wind):
    print(f"-----{cfg.name}-----")
    X_test_t = torch.from_numpy(cfg.X_test).float().to(device)
    y_test_t = torch.from_numpy(cfg.y_test).float().to(device)

    cfg.model.eval()
    with torch.inference_mode():
        y_pred_t = cfg.model(X_test_t)    # shape (N, n_features)

    y_pred = cfg.scaler.inverse_transform(y_pred_t.cpu().numpy())
    y_true = cfg.scaler.inverse_transform(y_test_t.cpu())

    all_y_pred.append(y_pred)
    all_y_true.append(y_true)

    # 1) MSE e RMSE por feature
    mse  = ((y_true - y_pred)**2).mean(axis=0)
    rmse = np.sqrt(mse)

    # 2) Desvio-padrão real de cada feature
    std  = y_true.std(axis=0)

    # 3) Erro normalizado (NRMSE = RMSE / std)
    nrmse = rmse / std

    # 4) R² por feature (quanto da variância o modelo explica)
    r2    = [r2_score(y_true[:,i], y_pred[:,i])
            for i in range(len(cfg.y_features))]

    cfg.metrics_per_feature = pd.DataFrame({
        'feature': cfg.y_features,
        'std':     std,
        'MSE':     mse,
        'RMSE':    rmse,
        'NRMSE':   nrmse,
        'R2':      r2
    })
    print(cfg.metrics_per_feature)

# %%
# Corrigir: concatenar listas em arrays
y_true_all = np.concatenate(all_y_true, axis=1)  # (N, total_features)
y_pred_all = np.concatenate(all_y_pred, axis=1)  # (N, total_features)

# Agora calcular as métricas
mse_total = ((y_true_all - y_pred_all) ** 2).mean()
rmse_total = np.sqrt(mse_total)
std_total = y_true_all.std()
nrmse_total = rmse_total / std_total
r2_total = r2_score(y_true_all, y_pred_all)

global_metrics = pd.DataFrame({
    'MSE':   [mse_total],
    'RMSE':  [rmse_total],
    'NRMSE': [nrmse_total],
    'R2':    [r2_total]
})
print(global_metrics)

# %%
#1) Concatenar y_true e y_pred de todos os modelos
all_y_true = []
all_y_pred = []
for cfg in (thermo_hydro, precip, wind):
   # predição em escala real
   X_test_t   = torch.from_numpy(cfg.X_test).float().to(device)
   y_pred_t   = cfg.model(X_test_t).detach().cpu().numpy()
   y_pred_inv = cfg.scaler.inverse_transform(y_pred_t)
   y_true_inv = cfg.scaler.inverse_transform(cfg.y_test)
   all_y_true.append(y_true_inv)
   all_y_pred.append(y_pred_inv)

y_true_all = np.concatenate(all_y_true, axis=1)
y_pred_all = np.concatenate(all_y_pred, axis=1)

# 2) Lista completa de features, na mesma ordem
features_all = (
   thermo_hydro.y_features +
   precip.y_features +
   wind.y_features
)

# 3) Índice de teste considerando window_size de cada cfg
#    (aqui todos usam o mesmo, então pegamos do primeiro)
ws = thermo_hydro.window_size
n = len(X_train_scaled)

if isinstance(df.index, pd.DatetimeIndex):
   idx_test = df.index[n + ws : n + ws + len(y_true_all)]
else:
   idx_test = np.arange(n + ws, n + ws + len(y_true_all))

# 4) DataFrames longos para plotnine
df_true = pd.DataFrame(y_true_all, index=idx_test, columns=features_all)
df_pred = pd.DataFrame(y_pred_all, index=idx_test, columns=features_all)

# recorte para visualização
start, end = 0, 200
slice_true = df_true.iloc[start:end]
slice_pred = df_pred.iloc[start:end]

df_plot = (
   pd.concat([
       slice_true.assign(type="Real"),
       slice_pred.assign(type="Previsto")
   ])
   .reset_index()
   .melt(
       id_vars=['index', 'type'],
       value_vars=features_all,
       var_name='feature',
       value_name='value'
   )
)

# 5) Monta e exibe o gráfico
plot = (
   ggplot(df_plot, aes(x='index', y='value', color='type'))
   + geom_line()
   + facet_wrap('~feature', scales='free_y', ncol=1)
   + labs(
       title="Real vs Previsto (todos os modelos)",
       x="Timestamp (ou amostra)",
       y="Valor",
       color=""
   )
   + theme_bw()
   + theme(figure_size=(12, 20))
)
# %%
def get_next_number(folder):
    numbers = os.listdir(folder)
    next_number = max(int(number) for number in numbers) + 1
    os.mkdir(f"{folder}/{next_number}")
    return next_number

# Save
next_number = get_next_number("results/main")
filename = f"results/main/{next_number}/comparation.png"
plot.save(filename,dpi=300)
print(f"Saved to {filename}")

# %%

for cfg in (thermo_hydro, precip, wind):
    results = {
        "features_X":          common_features,
        "features_y":          cfg.y_features,
        "train_size":          train_size,
        "window_size":         cfg.window_size,
        "input_size":          cfg.input_size,
        "output_size":         cfg.output_size,
        "hidden_size":         cfg.hidden_size,
        "num_layers":          cfg.num_layers,
        "learning_rate":       cfg.lr,
        "batch_size":          cfg.batch_size,
        "epochs":              cfg.epochs,
        "global_metrics":      global_metrics.to_dict(orient='records'),
        "final_train_loss":    cfg.train_losses[-1],
        "train_losses":        cfg.train_losses,
        "metrics_by_feature":  cfg.metrics_per_feature.to_dict(orient='records'),
    }

    filename = f"results/main/{next_number}/result_{cfg.name}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print("📊 Todos os resultados foram gravados em results.json")

# %%
for cfg in (thermo_hydro, precip, wind):
    filename = f"results/main/{next_number}/model_{cfg.name}.h5"
    torch.save(cfg.model.state_dict(), filename)
