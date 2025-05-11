import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

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

# Dicionário para armazenar os resultados finais
best_params = {}
results = {
    'Feature': [],
    'MSE': [],
    'RMSE': [],
    'R2': [],
    'NRMSE': [],
    'STD': []
}

# Para cada feature alvo
for target in features_y:
    print(f"\n{'='*50}")
    print(f"Otimizando modelo para prever: {target}")
    print(f"{'='*50}")

    # Remover o target atual das features se estiver presente
    X_features = [f for f in features_X if f != target]

    # Separar os dados em X e y
    X = df[X_features]
    y = df[target]

    # Split dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir a função objetivo para otimização com Optuna
    def objective(trial):
        # Parâmetros a serem otimizados
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42
        }

        # Inicializar o modelo com os parâmetros sugeridos
        model = XGBRegressor(**param)

        # Validação cruzada para evitar overfitting
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train,
                                   scoring='neg_mean_squared_error',
                                   cv=kf, n_jobs=-1)

        # Retornar a média negativa do erro quadrado (que será minimizada)
        return np.mean(-cv_scores)  # Optuna minimiza o objetivo

    # Criar o estudo Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # 50 tentativas por padrão, ajuste conforme necessário

    # Obter os melhores parâmetros
    best_trial = study.best_trial
    best_params[target] = best_trial.params
    print(f"Melhores parâmetros para {target}:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Treinar o modelo final com os melhores parâmetros
    final_model = XGBRegressor(**best_params[target])
    final_model.fit(X_train, y_train)

    # Fazer predições
    y_pred = final_model.predict(X_test)

    # Calcular métricas finais
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    nrmse_val = nrmse(y_test, y_pred)
    std = np.std(y_test - y_pred)

    # Armazenar resultados
    results['Feature'].append(target)
    results['MSE'].append(mse)
    results['RMSE'].append(rmse)
    results['R2'].append(r2)
    results['NRMSE'].append(nrmse_val)
    results['STD'].append(std)

    print(f"\nMétricas finais para {target}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"NRMSE: {nrmse_val:.4f}")
    print(f"STD: {std:.4f}")

    # Feature importance
    importance = final_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))

    # Visualizações Optuna (opcional - salvar gráficos)
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(f"optuna_history_{target}.png")

        fig2 = plot_param_importances(study)
        fig2.write_image(f"param_importance_{target}.png")
    except:
        print("Não foi possível criar as visualizações Optuna (talvez bibliotecas extras sejam necessárias)")

# Criar DataFrame de resultados
results_df = pd.DataFrame(results)

# Mostrar tabela de resultados
print("\nTabela de resultados finais após otimização:")
print(results_df)

# Salvar resultados
results_df.to_csv('resultados_otimizados_xgboost.csv', index=False)

# Salvar os melhores parâmetros em um arquivo
with open('melhores_parametros_xgboost.txt', 'w') as f:
    for target, params in best_params.items():
        f.write(f"Melhores parâmetros para {target}:\n")
        for key, value in params.items():
            f.write(f"    {key}: {value}\n")
        f.write("\n")

# Visualizar resultados finais
plt.figure(figsize=(12, 8))
sns.barplot(x='Feature', y='R2', data=results_df)
plt.title('R² por Feature após Otimização')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('r2_otimizado_por_feature.png')

plt.figure(figsize=(12, 8))
sns.barplot(x='Feature', y='RMSE', data=results_df)
plt.title('RMSE por Feature após Otimização')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rmse_otimizado_por_feature.png')

# Criar tabela comparativa formatada bonita
styled_results = results_df.style.background_gradient(cmap='coolwarm', subset=['R2'])\
                                .background_gradient(cmap='coolwarm_r', subset=['MSE', 'RMSE', 'NRMSE', 'STD'])\
                                .format({'MSE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}',
                                         'NRMSE': '{:.4f}', 'STD': '{:.4f}'})

# Salvar como HTML
styled_results.to_html('resultados_otimizados_xgboost.html')

print("\nOtimização concluída! Resultados e visualizações foram salvos nos arquivos correspondentes.")
