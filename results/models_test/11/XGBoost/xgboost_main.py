import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# Assumindo que você tenha um DataFrame chamado 'df' com todas essas colunas
# Se você não tiver o DataFrame, você precisará carregá-lo:
df = pd.read_csv('data/cleaned_data.csv')

# Função para calcular NRMSE (Normalized RMSE)
def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    if range_y == 0:
        return np.nan  # Evitar divisão por zero
    return rmse / range_y

# Dicionário para armazenar os resultados
results = {
    'feature': [],
    'MSE': [],
    'RMSE': [],
    'R2': [],
    'NRMSE': [],
    'std': []
}

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

# Para cada feature alvo
for target in features_y:
    print(f"\nTreinando modelo para prever: {target}")

    # Remover o target atual das features se estiver presente
    X_features = [f for f in features_X if f != target]

    # Separar os dados em X e y
    X = df[X_features]
    y = df[target]

    # Split dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo XGBoost
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Fazer predições
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)
    std = np.std(y_test - y_pred)

    # Armazenar resultados
    results['feature'].append(target)
    results['MSE'].append(mse)
    results['RMSE'].append(rmse)
    results['R2'].append(r2)
    results['NRMSE'].append(nrmse_val)
    results['std'].append(std)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")
    print(f"std: {std:.4f}")

    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance.head(10))  # Top 10 features

# Criar DataFrame de resultados
results_df = pd.DataFrame(results)

# Mostrar tabela de resultados
print("\nTabela de resultados:")
print(results_df)

# Visualizar resultados
plt.figure(figsize=(12, 8))
sns.barplot(x='feature', y='R2', data=results_df)
plt.title('R² por Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('r2_por_feature.png')

plt.figure(figsize=(12, 8))
sns.barplot(x='feature', y='RMSE', data=results_df)
plt.title('RMSE por Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rmse_por_feature.png')

# Criar tabela comparativa formatada bonita
styled_results = results_df.style.background_gradient(cmap='coolwarm', subset=['R2'])\
                                .background_gradient(cmap='coolwarm_r', subset=['MSE', 'RMSE', 'NRMSE', 'std'])\
                                .format({'MSE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}',
                                         'NRMSE': '{:.4f}', 'std': '{:.4f}'})

# Salvar como HTML ou imagem
styled_results.to_html('resultados_xgboost.html')
