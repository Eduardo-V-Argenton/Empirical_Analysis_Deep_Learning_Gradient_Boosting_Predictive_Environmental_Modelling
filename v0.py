
# %%
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, vonmises, gamma,lognorm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme
from torch.nn.modules import PReLU

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
# Configurações iniciais
np.random.seed(42)
start_date = '2025-01-01'
end_date = '2035-12-31'
freq = 'h'
datetime_index = pd.date_range(start=start_date, end=end_date, freq=freq)

# Parâmetros baseados em padrões do litoral paulista [[3]][[8]]
TEMP_MEAN = 22  # °C
TEMP_AMP_ANNUAL = 8  # Variação anual
TEMP_AMP_DAILY = 5   # Variação diária

HUMIDITY_BASE = 80   # % (média anual)
PRECIP_RATE = 0.2    # Probabilidade horária de chuva (ajustado sazonalmente)

WIND_SPEED_SHAPE = 2.0  # Forma da distribuição de Weibull
WIND_SPEED_SCALE = 8.0  # Escala (km/h)

# Funções auxiliares
def generate_temperature(index):
    """Gera temperatura com ciclos diurno e sazonal [[8]]"""
    times = np.arange(len(index))
    annual_cycle = TEMP_AMP_ANNUAL * np.sin(2 * np.pi * (times/8760))
    daily_cycle = TEMP_AMP_DAILY * np.sin(2 * np.pi * (times/24))
    noise = np.random.normal(0, 2, len(index))
    return TEMP_MEAN + annual_cycle + daily_cycle + noise

def generate_humidity(temp, precip):
    """Umidade relativa baseada em temperatura e precipitação [[1]]"""
    base = HUMIDITY_BASE - 0.5 * temp  # Relação inversa com temperatura
    rain_effect = np.where(precip > 0, 15, 0)  # Aumento de 15% durante chuva
    return np.clip(base + rain_effect + np.random.normal(0, 5, len(temp)), 30, 100)

def generate_wind():
    """Velocidade e direção do vento (costa brasileira: predominância de E/SE) [[4]]"""
    speed = weibull_min.rvs(WIND_SPEED_SHAPE, scale=WIND_SPEED_SCALE, size=len(datetime_index))
    direction = vonmises.rvs(loc=np.radians(120), kappa=2, size=len(datetime_index))  # 120° = ESE
    return np.clip(speed, 0, 50), np.degrees(direction) % 360

def generate_precipitation(index):
    """
    Precipitação com padrão sazonal (mais chuva no verão)
    - Usa distribuição log-normal para melhor capturar eventos extremos.
    - Limite máximo ajustado para 200 mm (valores realistas para o litoral).
    """
    monthly_probs = index.month.map({
        1:0.15, 2:0.2, 3:0.25, 4:0.18, 5:0.12, 6:0.08,
        7:0.05, 8:0.06, 9:0.1, 10:0.15, 11:0.2, 12:0.25
    })

    precip = np.zeros(len(index))
    for i, prob in enumerate(monthly_probs):
        if np.random.rand() < prob:
            # Parâmetros da log-normal (shape=1.2, loc=0, scale=10)
            precip[i] = lognorm.rvs(s=1.2, scale=10) * 8  # Multiplicador para intensidade realista

    return np.clip(precip, 0, 200)  # Limite máximo aumentado para 200 mm


def generate_soil_vars(temp, precip):
    """Umidade e temperatura do solo [[2]]"""
    soil_moisture = np.zeros(len(temp))
    soil_temp = np.zeros(len(temp))
    for t in range(1, len(temp)):
        # Dinâmica de umidade do solo
        evap = 0.1 * temp[t] if temp[t] > 15 else 0.05 * temp[t]
        soil_moisture[t] = soil_moisture[t-1] + precip[t] - evap
        soil_moisture[t] = np.clip(soil_moisture[t], 10, 100)  # % de saturação

        # Temperatura do solo (resposta amortecida)
        soil_temp[t] = 0.8 * soil_temp[t-1] + 0.2 * temp[t]
    return soil_moisture, soil_temp

# Geração dos dados
temp = generate_temperature(datetime_index)
precip = generate_precipitation(datetime_index)
humid = generate_humidity(temp, precip)
wind_speed, wind_dir = generate_wind()
soil_moist, soil_temp = generate_soil_vars(temp, precip)

# Criação do DataFrame
data = pd.DataFrame({
    'datetime': datetime_index,
    'temperature': np.round(temp, 1),
    'humidity': np.round(humid).astype(int),
    'wind_speed_kmh': np.round(wind_speed, 1),
    'wind_direction_deg': np.round(wind_dir).astype(int),
    'precipitation_mm': np.round(precip, 1),
    'soil_moisture_pct': np.round(soil_moist).astype(int),
    'soil_temperature': np.round(soil_temp, 1)
})

data['hour'] = datetime_index.hour
data['month'] = datetime_index.month

# Transformação cíclica para hora
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Transformação cíclica para mês
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Remove colunas temporárias
data.drop(['hour', 'month'], axis=1, inplace=True)
data.head()

# %%

#data.set_index('datetime', inplace=True)
#data.head()
# %%
# %%
print(data.isnull().sum())  # Verifique colunas com NaN

# Preencha valores ausentes (usando forward-fill e backward-fill)
data.fillna(method='ffill', inplace=True)  # Preenche com o valor anterior
data.fillna(method='bfill', inplace=True)  # Preenche com o próximo valor (caso o primeiro valor seja NaN)

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    FunctionTransformer,
    PowerTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Transformação para direção do vento (circular)
def transform_wind_direction(X):
    wind_dir_rad = np.deg2rad(X)  # Converte graus para radianos
    return np.column_stack([np.sin(wind_dir_rad), np.cos(wind_dir_rad)])

# Configurar ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        # Features cíclicas (já pré-processadas)
        ('hour_sin', 'passthrough', ['hour_sin']),
        ('hour_cos', 'passthrough', ['hour_cos']),
        ('month_sin', 'passthrough', ['month_sin']),
        ('month_cos', 'passthrough', ['month_cos']),
        ('wind_dir_sin', FunctionTransformer(lambda x: np.sin(x * (2 * np.pi / 360))), ['wind_direction_deg']),
        ('wind_dir_cos', FunctionTransformer(lambda x: np.cos(x * (2 * np.pi / 360))), ['wind_direction_deg']),
        ('temperature', StandardScaler(), ['temperature']),
        ('humidity', MinMaxScaler(feature_range=(0, 1)), ['humidity']),
        ('wind_speed', Pipeline([
            ('log', FunctionTransformer(np.log1p)),
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ]), ['wind_speed_kmh']),
        ('precipitation', Pipeline([
            ('log', FunctionTransformer(np.log1p)),  # Log para reduzir outliers
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ]), ['precipitation_mm']),
        ('soil_moisture', MinMaxScaler(feature_range=(0, 1)), ['soil_moisture_pct']),
        ('soil_temp', StandardScaler(), ['soil_temperature'])
    ],
    remainder='drop'
)

# %%
train_size = int(0.8 * len(data))  # 80% para treino
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# %%
X_train = train_data.drop(columns=['datetime'])
y_train = train_data['temperature']
preprocessor.fit(X_train)

X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(test_data.drop(columns=['datetime']))
y_test = test_data['temperature']
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # (n_amostras, 1)

# Normaliza y_test com o mesmo scaler (SEM fit!)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))        # (n_amostras, 1)
X_train_scaled.shape

# %%

def create_sequences(data, targets, seq_length):
    sequences = []
    target_seq = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        target_seq.append(targets[i+seq_length])
    return torch.FloatTensor(np.array(sequences)), torch.FloatTensor(np.array(target_seq))

seq_length = 24
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

print(f"Formato do treino: {X_train_seq.shape} {y_train_seq.shape}")  # (amostras, look_back, características)
print(f"Formato do teste: {X_test_seq.shape} {y_test_seq.shape}")
# %%

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        errors = preds - targets
        loss = torch.log(torch.cosh(errors + 1e-12))  # Evita log(0)
        return torch.mean(loss)
# %%
# 3. Definir modelo LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)  # hn: hidden state, cn: cell state
        x = self.linear(x[:, -1, :])  # Pegar apenas a última saída
        return x

# %%
# Modelo com atenção:
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # Novas features
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=0.2
        )
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)  # Self-attention
        return self.linear(x[:, -1, :])
# %%
model = EnhancedLSTM(input_size=12, hidden_size=256, num_layers=2)
model.to(device)
criterion = LogCoshLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,       # Redução menos agressiva (50% da LR atual)
    patience=10,      # Dê mais tempo para o modelo após cada redução
    min_lr=1e-6,      # Defina um LR mínimo menor (ex: 1e-6)
)
# %%
X_train = torch.FloatTensor(X_train_seq)
y_train = torch.FloatTensor(y_train_seq) # or LongTensor for classification

dataset = torch.utils.data.TensorDataset(X_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# %%

# 4. Treinamento
epochs = 200
window_size = 20  # Tamanho da janela para a média móvel
losses = []                # Lista para armazenar as losses de cada época
best_avg_loss = np.inf     # Melhor média móvel
best_loss = np.inf         # Melhor loss absoluta
epochs_since_best = 0      # Contador de épocas sem melhoria
losses_json=[]

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # Loop de batches
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    # Loss média da época (correto!)
    epoch_loss = epoch_loss / len(dataloader.dataset)
    losses.append(epoch_loss)  # Armazena a loss média da época

    # Atualiza o scheduler com a loss média da época
    scheduler.step(epoch_loss)

    # --- Média Móvel para Early Stopping ---
    if epoch >= window_size:
        avg_loss = np.mean(losses[-window_size:])  # Média das últimas 'window_size' épocas
    else:
        avg_loss = epoch_loss  # Para as primeiras épocas

    # Checkpoint do melhor modelo (baseado na média móvel)
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss
        best_loss = epoch_loss  # Salva também a melhor loss absoluta
        torch.save(model.state_dict(), 'best_model.pth')  # Salva os pesos
        epochs_since_best = 0
        print(f'New Best Avg Loss: {best_avg_loss:.5f}, Best Loss: {best_loss:.5f}')
    else:
        epochs_since_best += 1

    # Early stopping baseado na média móvel
    if epochs_since_best > 20:
        print(f'Early stopping at epoch {epoch}')
        break

    # Log a cada 5 épocas
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        losses_json.append(epoch_loss)

# %%
rounded_loss =[round(loss,4) for loss in losses_json]
json.dumps(rounded_loss)
# %%
print("Shapes originais:")
print(f"X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}, y_test: {y_test.shape}")

print("\nShapes após create_sequences:")
print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")
# %%
model.eval()
predictions = []

# Configurações para economia de memória
batch_size = 32  # Reduza conforme necessário para sua GPU
with torch.inference_mode():
    # Cria DataLoader para processar em batches
    test_dataset = torch.utils.data.TensorDataset(X_test_seq)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch in test_loader:
        inputs = batch[0].to(device)

        # Faz as previsões
        batch_pred = model(inputs)

        # Move as previsões para CPU e libera memória imediatamente
        predictions.append(batch_pred.cpu().detach())
        del inputs, batch_pred
        torch.cuda.empty_cache()  # Limpa memória não utilizada

# Concatena todas as previsões
predictions = torch.cat(predictions).numpy()

# Libera memória dos tensores grandes
del X_test_seq, test_loader
torch.cuda.empty_cache()


y_test_np = y_test_seq  # Assume que y_test é um DataFrame/Series
predictions_original = scaler_y.inverse_transform(predictions)
y_test_np = scaler_y.inverse_transform(y_test_seq)
# Comparar dimensões
print("Shape das previsões:", predictions.shape)
print("Shape do y_test:", y_test_np.shape)  # Devem ser compatíveis!

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
r2 = r2_score(y_test_np, predictions_original)
mse = mean_squared_error(y_test_np, predictions_original)
mae = mean_absolute_error(y_test_np, predictions_original)
mape = mean_absolute_percentage_error(y_test_np, predictions_original)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2: {r2:.4f}")
# %%
# Resíduos vs Tempo
residuals = y_test_np - predictions_original
plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.savefig('residuals_over_time.png')
plt.axhline(0, color='red', linestyle='--')
plt.title("Resíduos ao Longo do Tempo")

# ACF dos Resíduos
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=40)
plt.savefig('acf_residuos.png')

# %%
import matplotlib.pyplot as plt

# Amostrar 100 pontos aleatórios
sample_indices = np.random.choice(len(y_test_np), 100, replace=False)
y_test_sample = y_test_np[sample_indices]
predictions_sample = predictions_original[sample_indices]

# Plotar valores reais vs. previsões
plt.figure(figsize=(10, 6))
plt.plot(y_test_sample, label='Valores Reais (y_test)', alpha=0.7)
plt.plot(predictions_sample, label='Previsões', linestyle='--', alpha=0.7)
plt.xlabel("Amostras")
plt.ylabel("Valor")
plt.legend()
plt.title("Comparação entre Valores Reais e Previsões (Amostra)")
plt.show()
plt.savefig('real_vs_predicted.png')
# %%
import matplotlib.pyplot as plt

plt.plot(losses, label='Loss por Época')
plt.plot([np.mean(losses[max(0, i-window_size):i+1]) for i in range(len(losses))],
         label=f'Média Móvel ({window_size} épocas)', color='red')
plt.legend()
plt.xlabel('Época')
plt.ylabel('Loss')
plt.show()
plt.savefig('loss_over_epochs.png')

# %%
# Exemplo para a feature 'temperature' após StandardScaler
import seaborn as sns

plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 0])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'hour_sin' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 1])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'hour_cos' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 2])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'month_sin' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 3])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'month_cos' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 4])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'wind_direction_deg_sin' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 5])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'wind_direction_deg_cos' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 6])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'temperature' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 7])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'humidity' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 8])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'wind_speed_km/h' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 9])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'preciption' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 10])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'soil_moisture' (Padronizada)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train_scaled[:, 11])  # Substitua 0 pelo índice da feature
plt.title("Distribuição da Feature 'soil_temperature' (Padronizada)")
plt.show()
