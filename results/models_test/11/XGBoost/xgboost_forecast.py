# %%
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from joblib import load, dump
from datetime import datetime, timedelta
from plotnine import ggplot, aes, geom_line, labs, theme_bw, facet_wrap, scale_x_continuous
import matplotlib.pyplot as plt
import os

# Configurações
MODEL_DIR = 'results/xgboost'  # Diretório onde serão salvos os modelos XGBoost
DATA_PATH = '../../../../data/cleaned_data.csv'
STEPS = 72  # Número de passos de previsão

# Lista de features
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

# Parâmetros do XGBoost
params = {
    'n_estimators': 372,
    'max_depth': 9,
    'learning_rate': 0.15038988832914096,
    'subsample': 0.9550521763800149,
    'colsample_bytree': 0.9697343861433267,
    'gamma': 0.0597224858297457,
    'min_child_weight': 10,
    'reg_alpha': 2.877497120964207,
    'reg_lambda': 2.722809408005546
}

# Criar diretório para os modelos se não existir
os.makedirs(MODEL_DIR, exist_ok=True)

# %%
# Carregar e preparar os dados
df = pd.read_csv(DATA_PATH)
print(f"Data loaded with shape: {df.shape}")

# %%
# Função para treinar modelos XGBoost para cada feature target
def train_xgboost_models():
    models = {}

    for target in features_y:
        print(f"\nTreinando modelo para prever: {target}")

        # Remover o target atual das features se estiver presente
        X_features = [f for f in features_X if f != target]

        # Separar os dados em X e y
        X = df[X_features]
        y = df[target]

        # Treinar o modelo XGBoost
        model = XGBRegressor(**params)
        model.fit(X, y)

        # Salvar o modelo
        model_path = f"{MODEL_DIR}/{target}_model.json"
        model.save_model(model_path)
        print(f"Modelo salvo em: {model_path}")

        # Armazenar o modelo em memória para uso imediato
        models[target] = model

    return models

# Verificar se precisamos treinar os modelos
def load_or_train_models():
    models = {}
    all_exist = True

    for target in features_y:
        model_path = f"{MODEL_DIR}/{target}_model.json"
        if not os.path.exists(model_path):
            all_exist = False
            break

    if all_exist:
        print("Carregando modelos existentes...")
        for target in features_y:
            model_path = f"{MODEL_DIR}/{target}_model.json"
            model = XGBRegressor()
            model.load_model(model_path)
            models[target] = model
    else:
        print("Treinando novos modelos...")
        models = train_xgboost_models()

    return models

# %%
# Carregar ou treinar modelos
models = load_or_train_models()

# %%
# Função para fazer previsões com o XGBoost
def make_forecasts(steps=STEPS):
    predictions = []

    # Obter o último timestamp do dataframe para continuar a partir dele
    last_timestamp = pd.to_datetime(df["Timestamp"].iloc[-1]) if "Timestamp" in df.columns else datetime.now().replace(minute=0, second=0, microsecond=0)

    # Último delta_t conhecido (intervalo de tempo em segundos entre registros)
    try:
        last_delta_t = df["delta_t"].iloc[-1]
        if pd.isnull(last_delta_t):
            last_delta_t = 3600  # 1 hora em segundos
    except (KeyError, IndexError):
        last_delta_t = 3600

    # Pegar o último registro para iniciar as previsões
    current_data = df.iloc[-1].copy()

    for step in range(steps):
        # Simular o próximo timestamp
        sim_time = last_timestamp + timedelta(hours=step + 1)
        hour = sim_time.hour
        doy = sim_time.timetuple().tm_yday
        dow = sim_time.weekday()  # 0 é segunda-feira, 6 é domingo

        # Calcular features temporais
        ts_unix = sim_time.timestamp()

        # Para ts_norm, usar a última proporção conhecida
        try:
            ts_norm_ratio = df["ts_norm"].iloc[-1] / df["ts_unix"].iloc[-1]
            ts_norm = ts_unix * ts_norm_ratio
        except (KeyError, IndexError, ZeroDivisionError):
            ts_norm = ts_unix / 86400  # normalização simples

        # Atualizar features temporais no registro atual
        current_data["ts_unix"] = ts_unix
        current_data["ts_norm"] = ts_norm
        current_data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        current_data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        current_data["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        current_data["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        current_data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        current_data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        current_data["delta_t"] = last_delta_t

        # Para delta_t_norm, seguir a mesma lógica de ts_norm
        try:
            delta_t_norm_ratio = df["delta_t_norm"].iloc[-1] / df["delta_t"].iloc[-1]
            current_data["delta_t_norm"] = current_data["delta_t"] * delta_t_norm_ratio
        except (KeyError, IndexError, ZeroDivisionError):
            current_data["delta_t_norm"] = current_data["delta_t"] / 86400

        # Dicionário para armazenar os resultados da previsão
        row = {'Step': step + 1, 'Timestamp': sim_time.strftime('%Y-%m-%d %H:%M:%S')}

        # Fazer previsões para cada target
        for target in features_y:
            # Preparar input para este target específico
            X_features = [f for f in features_X if f != target]
            X_input = pd.DataFrame([current_data[X_features]], columns=X_features)

            # Fazer previsão
            pred_value = models[target].predict(X_input)[0]
            row[target] = pred_value

            # Atualizar o valor no registro atual para a próxima iteração
            current_data[target] = pred_value

        # Armazenar os resultados
        predictions.append(row)

    return pd.DataFrame(predictions)

# %%
# Executar previsões
forecast_df = make_forecasts(STEPS)

# Exibir resultados
print(forecast_df[['Step', 'Timestamp'] + features_y])

# Salvar CSV
forecast_df.to_csv(f'previsao_xgboost_proximas_{STEPS}_h.csv', index=False)

# %%
# Plot
plot_df = forecast_df.melt(id_vars=['Step', 'Timestamp'], value_vars=features_y,
                         var_name='Variável', value_name='Valor')

plot = (
    ggplot(plot_df, aes(x='Step', y='Valor', color='Variável'))
    + geom_line(size=1)
    + facet_wrap('~Variável', scales='free_y', ncol=2)
    + labs(title=f'Previsão XGBoost para as Próximas {STEPS} Horas',
         x='Horas à Frente', y='Valor Previsto')
    + theme_bw()
    + scale_x_continuous(breaks=range(0, STEPS + 1, 6))
)

plot.save('previsao_xgboost.png', dpi=300, height=8, width=10)
print(plot)

# %%
# Função para avaliar as previsões em dados de teste (opcional)
def evaluate_models(test_size=48):
    """
    Avalia os modelos XGBoost em um conjunto de teste.

    Args:
        test_size: Número de registros a serem usados para teste
    """
    if len(df) <= test_size:
        print("Dados insuficientes para avaliação")
        return

    # Separar dados de treino e teste
    train_data = df.iloc[:-test_size]
    test_data = df.iloc[-test_size:]

    results = {
        'feature': [],
        'MSE': [],
        'RMSE': [],
        'R2': [],
    }

    from sklearn.metrics import mean_squared_error, r2_score

    for target in features_y:
        print(f"Avaliando modelo para {target}...")

        # Remover o target atual das features
        X_features = [f for f in features_X if f != target]

        # Treinar modelo nos dados de treino
        X_train = train_data[X_features]
        y_train = train_data[target]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Avaliar no conjunto de teste
        X_test = test_data[X_features]
        y_test = test_data[target]

        y_pred = model.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Armazenar resultados
        results['feature'].append(target)
        results['MSE'].append(mse)
        results['RMSE'].append(rmse)
        results['R2'].append(r2)

        print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    print("\nResultados da avaliação:")
    print(results_df)

    # Salvar resultados
    results_df.to_csv('xgboost_evaluation_results.csv', index=False)

    return results_df

# Descomentar para executar avaliação (opcional)
# evaluation_results = evaluate_models()
