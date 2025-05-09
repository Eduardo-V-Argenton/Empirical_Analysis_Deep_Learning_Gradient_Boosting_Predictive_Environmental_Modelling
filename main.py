# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries,concatenate
from darts.models import RNNModel
from darts.metrics import mse, rmse, r2_score, mase, mae
from darts.dataprocessing.transformers import Scaler
import matplotlib.dates as mdates
from statsmodels.tsa.vector_ar.var_model import output

# Configurações de visualização
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# 1. Carregar e preparar os dados
target_columns = [
    'Temperature', 'Precipitation_log', 'Humidity', 'Wind_Speed_kmh',
    'Soil_Moisture', 'Soil_Temperature', 'Wind_Dir_Sin', 'Wind_Dir_Cos'
]

# Carregar os dados
df1 = pd.read_csv("data/cleaned/ground_station_2022-08-03_2022-12-17.csv")
df2 = pd.read_csv("data/cleaned/ground_station_2023-08-02_2024-01-11.csv")
df3 = pd.read_csv("data/cleaned/ground_station_2024-05-29_2024-09-16.csv")

# Converter para séries temporais do Darts
series1 = TimeSeries.from_dataframe(df1, time_col="Timestamp", value_cols=target_columns, freq='30min')
series2 = TimeSeries.from_dataframe(df2, time_col="Timestamp", value_cols=target_columns, freq='30min')
series3 = TimeSeries.from_dataframe(df3, time_col="Timestamp", value_cols=target_columns, freq='30min')

# Verificar valores ausentes
print("Valores ausentes em cada série:")
print("Series 1:", series1.to_dataframe().isna().sum())
print("Series 2:", series2.to_dataframe().isna().sum())
print("Series 3:", series3.to_dataframe().isna().sum())

# %%
# 2. Dividir em conjuntos de treino e validação
# Usar 80% dos dados da primeira série para treino e 20% para validação
series1_train, series1_val = series1.split_before(0.8)

# %%
# 3. Normalizar os dados
scaler = Scaler()
# Ajustar o scaler apenas nos dados de treino para evitar data leakage
scaler = scaler.fit(series1_train)
# Transformar todas as séries
series1_train_scaled = scaler.transform(series1_train)
series1_val_scaled = scaler.transform(series1_val)
series2_scaled = scaler.transform(series2)
series3_scaled = scaler.transform(series3)

# %%
# 4. Definir e treinar o modelo
# Definir parâmetros do modelo
model = RNNModel(
    model="GRU",
    input_chunk_length=48,
    hidden_dim=128,
    n_rnn_layers=1,
    n_epochs=100,
    random_state=42,
    batch_size=64,
    dropout=0.2,
    training_length=336,
    optimizer_kwargs={"lr": 0.00016},  # Definir taxa de aprendizado
)

# Treinar com todas as séries disponíveis
print("\nTreinando o modelo...")
model.fit(
    series=[series1_train_scaled],
    verbose=True
)

# %%
n = 256
forecasts = model.predict(n,series=series1_val_scaled[:-n])
# %%
if isinstance(forecasts, list):
    print(f"Historical_forecasts retornou uma lista de {len(forecasts)} séries temporais.")
    forecasts = concatenate(forecasts)
print(f"Formato das previsões: {len(forecasts)}")
print(f"Início da previsão: {forecasts.start_time()}")
print(f"Fim da previsão: {forecasts.end_time()}")
print(f"Início da validação: {series1_val_scaled[-n:].start_time()}")
print(f"Fim da validação: {series1_val_scaled[-n:].end_time()}")
# %%
# 6. Avaliar métricas de performance
metrics = {}
series1_val_t = series1_val
forecasts_t = scaler.inverse_transform(forecasts)
print("Iniciando a verificação das séries temporais para constância...")
for target in target_columns:
    metrics[target] = {
        'MSE': mse(series1_val_t[target], forecasts_t[target]),
        'RMSE': rmse(series1_val_t[target], forecasts_t[target]),
        'MAE': mae(series1_val_t[target], forecasts_t[target]),
        'R2': r2_score(series1_val_t[target], forecasts_t[target]),
        # 'MASE': mase(series1_val_t[target], forecasts_t[target], series1_train_scaled[target])
    }

# %%
# Converter métricas para DataFrame para fácil visualização
metrics_df = pd.DataFrame(metrics).T
print("\nMétricas de performance:")
print(metrics_df)

# %%
for target in target_columns:
    plt.figure()  # Cria uma nova figura para cada target
    series1_val_t[target][-n:].plot(label='Real')
    forecasts_t[target].plot(label='Previsto')
    plt.title(f'Previsão vs Real - {target}')
    plt.legend()
    plt.grid(True)
    plt.show()
