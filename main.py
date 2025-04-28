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
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme,facet_wrap,scale_color_manual
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from pandas.api.types import CategoricalDtype
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler

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
start_date = '2024-01-01'
end_date = '2025-12-31'
freq = '5min'
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
    times = np.arange(len(index))

    samples_per_day = 24 * 60 // 5  # 288 amostras por dia
    samples_per_year = samples_per_day * 365  # 105120 (não considerando ano bissexto)

    annual_cycle = TEMP_AMP_ANNUAL * np.sin(2 * np.pi * times / samples_per_year)
    daily_cycle = TEMP_AMP_DAILY * np.sin(2 * np.pi * times / samples_per_day)

    noise = np.random.normal(0, 0.5, len(index))  # ruído mais suave
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
df = pd.DataFrame({
    'Timestamp': datetime_index,
    'Precipitation': np.round(precip, 1),
    'Temperature': np.round(temp, 1),
    'Humidity': np.round(humid).astype(int),
    'Wind_Speed_kmh': np.round(wind_speed, 1),
    'Wind_Direction': np.round(wind_dir).astype(int),
    'Soil_Moisture': np.round(soil_moist).astype(int),
    'Soil_Temperature': np.round(soil_temp, 1)
})

# %%
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
features = ['Temperature', 'Precipitation', 'Humidity', 'Wind_Speed_kmh',
            'Soil_Moisture', 'Soil_Temperature', 'hour',
            'Wind_Dir_Sin', 'Wind_Dir_Cos']

X = df[features]

# Separar treino e teste
train_size = int(0.8 * len(X))  # 80% treino, 20% teste
X_train_data = X.iloc[:train_size]
X_test_data = X.iloc[train_size:]

# Escalonar com MinMaxScaler apenas no treino
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

# AGORA, vamos construir X e y para LSTM (com janela de tempo)

def create_sequences(X_scaled, window_size):
    Xs, ys = [], []
    for i in range(len(X_scaled) - window_size):
        Xs.append(X_scaled[i:i+window_size])    # sequência
        ys.append(X_scaled[i+window_size])       # próximo passo
    return np.array(Xs), np.array(ys)

# Define o tamanho da janela (ex: 10 passos anteriores)
window_size = 10

X_train, y_train = create_sequences(X_train_scaled, window_size)
X_test, y_test = create_sequences(X_test_scaled, window_size)

# %%
X_train.shape, y_train.shape

# %%
