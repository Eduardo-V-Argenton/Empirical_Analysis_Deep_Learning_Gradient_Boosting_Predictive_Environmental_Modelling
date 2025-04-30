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
features_y = ['Temperature', 'Precipitation', 'Humidity', 'Wind_Speed_kmh',
            'Soil_Moisture', 'Soil_Temperature',
            'Wind_Dir_Sin', 'Wind_Dir_Cos']

features_X = features_y + ['hour']
X = df[features_X]
y = df[features_y]
# Separar treino e teste
n = len(X)
n_trainval = int(0.8 * n)
n_test     = n - n_trainval
n_train    = int(0.8 * n_trainval)
n_val      = n_trainval - n_train

X_train, X_val, X_test = (
    X[:n_train],
    X[n_train:n_train + n_val],
    X[n_trainval:]
)
y_train, y_val, y_test = (
    y[:n_train],
    y[n_train:n_train + n_val],
    y[n_trainval:]
)

# 4) Escalonamento (fit apenas em treino)
X_scaler = MinMaxScaler((0,1)); X_scaler.fit(X_train)
y_scaler = MinMaxScaler((0,1)); y_scaler.fit(y_train)

X_train_s = X_scaler.transform(X_train)
X_val_s   = X_scaler.transform(X_val)
X_test_s  = X_scaler.transform(X_test)

y_train_s = y_scaler.transform(y_train)
y_val_s   = y_scaler.transform(y_val)
y_test_s  = y_scaler.transform(y_test)

# AGORA, vamos construir X e y para LSTM (com janela de tempo)

def create_sequences(Xs, ys, window_size):
    X_seq, y_seq = [], []
    for i in range(len(Xs) - window_size):
        X_seq.append(Xs[i:i+window_size])
        y_seq.append(ys[i+window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 12
X_tr, y_tr = create_sequences(X_train_s, y_train_s, window_size)
X_va, y_va = create_sequences(X_val_s,   y_val_s,   window_size)
X_te, y_te = create_sequences(X_test_s,  y_test_s,  window_size)

# %%
# Hiperparâmetros
input_size  = X_tr.shape[2]  # n_features
output_size  = y_tr.shape[1]  # n_features
hidden_size = 32
num_layers  = 1
lr          = 1e-3
batch_size  = 32
epochs      = 100

# %%
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                        torch.from_numpy(y_tr).float()),
                          batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va).float(),
                                        torch.from_numpy(y_va).float()),
                          batch_size, shuffle=False)

# %%
# Modelo
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
model = LSTM(input_size, hidden_size, num_layers, output_size)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
)
# %%

# %%
# Loop de treino
train_losses = []
val_losses = []
y_pred = []
y_true = []
best_avg_val_loss = float('inf')
epochs_since_best = 0
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb.to(device))            # (B, n_features)
        loss = criterion(pred, yb.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)

    # --- Validação ---
    model.eval()
    running_val = 0
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            running_val += criterion(pred, yb).item() * xb.size(0)
    avg_val = running_val / len(val_loader.dataset)
    val_losses.append(avg_val)

    scheduler.step(avg_val)

    # --- Checa melhoria na VALIDAÇÃO ---
    if avg_val < best_avg_val_loss:
        best_avg_val_loss = avg_val
        epochs_since_best = 0
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_since_best += 1
        # Early stopping on-demand (baseado na validação)
        if epochs_since_best > 20: # Ajuste a paciência conforme necessário
            print(f'Early stopping at epoch {epoch} due to validation loss stagnation.')
            break

    # --- Logging ---
    current_lr = optimizer.param_groups[0]['lr'] # Pega o LR atual
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | train_loss: {avg_loss:.5f} | val_loss: {avg_val:.5f} | LR: {current_lr:.1e}") # Adicionado LR

# %%
X_test_t = torch.from_numpy(X_te).float().to(device)
y_test_t = torch.from_numpy(y_te).float().to(device)

# %%
model.eval()
with torch.inference_mode():
    y_pred_t = model(X_test_t)    # shape (N, n_features)

y_pred = y_scaler.inverse_transform(y_pred_t.cpu().numpy())
y_true = y_scaler.inverse_transform(y_te)

# %%
# 1) MSE e RMSE por feature
mse  = ((y_true - y_pred)**2).mean(axis=0)
rmse = np.sqrt(mse)

# 2) Desvio-padrão real de cada feature
std  = y_true.std(axis=0)

# 3) Erro normalizado (NRMSE = RMSE / std)
nrmse = rmse / std

# 4) R² por feature (quanto da variância o modelo explica)
r2    = [r2_score(y_true[:,i], y_pred[:,i])
        for i in range(len(features_y))]

df = pd.DataFrame({
    'feature': features_y,
    'std':     std,
    'MSE':     mse,
    'RMSE':    rmse,
    'NRMSE':   nrmse,
    'R2':      r2
})
print(df)

# %%
mse = mean_squared_error(y_pred, y_true)
rmse = np.sqrt(mse)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# %%
# 2. Cria um index para as previsões: ele começa em train_size + window_size
train_val_split = int(0.8 * len(df))     # ponto onde começa o teste no df original
start = train_val_split + window_size
end   = start + len(y_true)
if isinstance(df.index, pd.DatetimeIndex):
    idx_test = df.index[start:end]
else:
    idx_test = np.arange(start, end)

# 3. DataFrames
df_true = pd.DataFrame(y_true, index=idx_test, columns=features_y)
df_pred = pd.DataFrame(y_pred, index=idx_test, columns=features_y)

# Escolha um recorte de, digamos, 200 pontos
start, end = 0, 200
slice_true = df_true.iloc[start:end]
slice_pred = df_pred.iloc[start:end]

# Monta DataFrame longo para plotnine
df_plot = (
    pd.concat([
        slice_true.assign(type="Real"),
        slice_pred.assign(type="Previsto")
    ])
    .reset_index()
    .melt(id_vars=['index','type'], value_vars=features_y,
          var_name='feature', value_name='value')
)

plot = (
    ggplot(df_plot, aes(x='index', y='value', color='type'))
    + geom_line()
    + facet_wrap('~feature', scales='free_y', ncol=1)
    + labs(
        title="Real vs Previsto (slice de teste)",
        x="Timestamp (ou amostra)",
        y="Valor",
        color=""
    )
    + theme_bw()
    + theme(figure_size=(12, 16))
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
results = {
    "features_X":          features_X,
    "features_y":          features_y,
    "train_size":          n,
    "window_size":         window_size,
    "input_size":          input_size,
    "output_size":         output_size,
    "hidden_size":         hidden_size,
    "num_layers":          num_layers,
    "learning_rate":       lr,
    "batch_size":          batch_size,
    "epochs":              epochs,
    "final_train_loss":    train_losses[-1],
    "train_losses":        train_losses,
    "lstm_mse":            mse,
    "lstm_rmse":           rmse,
}

filename = f"results/main/{next_number}/result.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=4)

print("📊 Todos os resultados foram gravados em results.json")

# %%
filename = f"results/main/{next_number}/model.h5"
torch.save(model.state_dict(), filename)
