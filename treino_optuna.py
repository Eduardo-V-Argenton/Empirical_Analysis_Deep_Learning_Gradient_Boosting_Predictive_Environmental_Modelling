# %%
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
from basic import Model, features_y, features_X
from functools import partial
import time
import gc

# %%
# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preparação dos dados - feito apenas uma vez fora da função objetivo
df = pd.read_csv('data/cleaned_data.csv')
X = df[features_X]
y = df[features_y]

# Separar treino e teste - feito uma única vez
n = len(X)
n_trainval = int(0.8 * n)
n_test = n - n_trainval
n_train = int(0.8 * n_trainval)
n_val = n_trainval - n_train

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

# Definir dados pré-processados globalmente para cada scaler
scalers = {
    'StandardScaler': (StandardScaler(), StandardScaler()),
    'MinMaxScaler': (MinMaxScaler(), MinMaxScaler()),
    'RobustScaler': (RobustScaler(), RobustScaler())
}

# Pré-processar dados para cada tipo de scaler
preprocessed_data = {}
for scaler_name, (X_scaler, y_scaler) in scalers.items():
    X_scaler.fit(X_train)
    y_scaler.fit(y_train)

    X_train_s = X_scaler.transform(X_train)
    X_val_s = X_scaler.transform(X_val)
    X_test_s = X_scaler.transform(X_test)

    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    preprocessed_data[scaler_name] = {
        'X_train_s': X_train_s,
        'X_val_s': X_val_s,
        'X_test_s': X_test_s,
        'y_train_s': y_train_s,
        'y_val_s': y_val_s,
        'y_test_s': y_test_s
    }
# %%
def create_sequences_vectorized(X, y, window_size):
    """Versão vetorizada e otimizada da função create_sequences"""
    n_samples = len(X) - window_size

    # Verificação de segurança: garantir que temos amostras suficientes
    if n_samples <= 0:
        return np.array([]), np.array([])

    X_seq = np.zeros((n_samples, window_size, X.shape[1]))

    for i in range(window_size):
        X_seq[:, i, :] = X[i:i+n_samples]

    y_seq = y[window_size:]

    return X_seq, y_seq

# Limite o tamanho do cache para evitar excesso de memória
MAX_CACHE_SIZE = 10
sequence_cache = {}

def get_sequences(scaler_name, window_size):
    """Recupera ou cria sequências com cache limitado"""
    cache_key = f"{scaler_name}_{window_size}"

    if cache_key in sequence_cache:
        return sequence_cache[cache_key]

    # Limitar o tamanho do cache
    if len(sequence_cache) >= MAX_CACHE_SIZE:
        # Remover o primeiro item (mais antigo) se o cache estiver cheio
        oldest_key = next(iter(sequence_cache))
        del sequence_cache[oldest_key]
        # Forçar coleta de lixo para liberar memória
        gc.collect()

    data = preprocessed_data[scaler_name]
    X_tr, y_tr = create_sequences_vectorized(data['X_train_s'], data['y_train_s'], window_size)
    X_va, y_va = create_sequences_vectorized(data['X_val_s'], data['y_val_s'], window_size)

    if len(X_tr) == 0 or len(X_va) == 0:
        return None, None, None, None

    sequence_cache[cache_key] = (X_tr, y_tr, X_va, y_va)
    return X_tr, y_tr, X_va, y_va
# %%
# Configurações para early stopping melhoradas
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        # Se val_loss for NaN ou inf, ative early stopping imediatamente
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.early_stop = True
            return self.early_stop

        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop
# %%
def objective(trial):
    # Adicionar timeout para cada trial
    start_time = time.time()
    MAX_TRIAL_TIME = 600  # 10 minutos em segundos

    try:
        # --- 1. Sugerir Hiperparâmetros ---
        window_size = trial.suggest_int('window_size', 6, 48, step=6)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        dropout_rate = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        bidirectional_lstm = trial.suggest_categorical('bidirectional', [True, False])
        loss_name = trial.suggest_categorical('loss', ['mse', 'mae', 'huber'])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        scaler_name = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])

        # Recuperar sequências pré-processadas
        X_tr, y_tr, X_va, y_va = get_sequences(scaler_name, window_size)

        if X_tr is None or len(X_tr) < 10:  # Verificação adicional para garantir dados suficientes
            raise optuna.exceptions.TrialPruned("Window size too large for data split.")

        # Reduzir workers e usar processo serial para evitar contention
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
            batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'),
            num_workers=0  # Reduzido para 0 para evitar problemas de paralelismo
        )

        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()),
            batch_size=batch_size*2, shuffle=False, pin_memory=(device.type == 'cuda'),
            num_workers=0  # Reduzido para 0 para evitar problemas de paralelismo
        )

        # Configuração do modelo e critério de perda
        input_size = X_tr.shape[2]
        output_size = y_tr.shape[1]

        model = Model(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional_lstm).to(device)

        if loss_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_name == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss()

        # Configuração do otimizador
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Adicionar scheduler para ajuste automático da taxa de aprendizado
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )

        # Configurar early stopping
        early_stopping = EarlyStopping(patience=5)
        best_avg_val_loss = float('inf')

        # --- 3. Loop de Treino e Validação ---
        max_epochs = 30  # Reduzido de 50 para 30 com early stopping

        for epoch in range(1, max_epochs + 1):
            # Verificar timeout
            if time.time() - start_time > MAX_TRIAL_TIME:
                print(f"Trial {trial.number} excedeu o tempo limite. Interrompendo.")
                raise optuna.exceptions.TrialPruned("Timeout excedido")

            # Treinamento
            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)

                # Verificar se a perda é NaN ou inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Encontrado NaN ou Inf em loss no trial {trial.number}. Interrompendo.")
                    raise optuna.exceptions.TrialPruned("NaN ou Inf encontrado")

                loss.backward()

                # Adicionar clipping de gradiente para estabilidade
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(train_loader.dataset)

            # Validação - feita a cada época para maior estabilidade
            model.eval()
            running_val = 0

            with torch.inference_mode():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss = criterion(pred, yb).item()

                    # Verificar NaN ou Inf
                    if np.isnan(val_loss) or np.isinf(val_loss):
                        print(f"Encontrado NaN ou Inf em validation no trial {trial.number}. Interrompendo.")
                        raise optuna.exceptions.TrialPruned("NaN ou Inf encontrado na validação")

                    running_val += val_loss * xb.size(0)

            avg_val = running_val / len(val_loader.dataset)

            if avg_val < best_avg_val_loss:
                best_avg_val_loss = avg_val

            # Ajustar learning rate baseado na performance
            scheduler.step(avg_val)

            # Checagem de early stopping
            if early_stopping(avg_val):
                print(f"Early stopping ativado no epoch {epoch} para trial {trial.number}")
                break

            # Reportar ao Optuna para pruning
            trial.report(avg_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Limpar memória
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return best_avg_val_loss

    except Exception as e:
        # Capturar qualquer exceção não tratada
        print(f"Erro no trial {trial.number}: {str(e)}")
        raise optuna.exceptions.TrialPruned(f"Exceção: {str(e)}")

# Configuração dos prunadores e samplers do Optuna para maior eficiência
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=5,
    interval_steps=1  # Reduzido para 1 para verificar mais frequentemente
)

study = optuna.create_study(
    direction='minimize',
    pruner=pruner,
    sampler=optuna.samplers.TPESampler(n_startup_trials=5)  # Reduzido para acelerar
)

# Número de trials reduzido para demonstração, mas com processos paralelos
n_trials = 30
n_jobs = 1  # Reduzido para 1 para evitar problemas de paralelismo
# %%
# Execute a otimização com timeout global
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=3600)  # 1 hora de timeout total

# Imprima os resultados
print("Melhor trial:")
trial = study.best_trial
print(f"  Valor (menor val_loss): {trial.value}")
print("  Parâmetros: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Salvar visualizações
try:
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('optimization_history.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('param_importances.png')
except:
    print("Não foi possível gerar visualizações. Verifique se plotly está instalado.")

# Salvar os melhores parâmetros
import json
with open('results_optuna.json', "w") as f:
    json.dump(trial.params, f, indent=4)

print("\nTreinando o modelo final com os melhores parâmetros...")
