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
import traceback
import json
import warnings
warnings.filterwarnings('ignore')

# %%
# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo em uso: {device}")

# %%
# Preparação dos dados - feito apenas uma vez fora da função objetivo
print("Carregando dados...")
df = pd.read_csv('data/cleaned_data.csv')
X = df[features_X]
y = df[features_y]

print(f"Dados carregados: {len(df)} amostras, {len(features_X)} features de entrada, {len(features_y)} features de saída")

# %%
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

print(f"Divisão dos dados:")
print(f"  - Treino: {n_train} amostras")
print(f"  - Validação: {n_val} amostras")
print(f"  - Teste: {n_test} amostras")

# %%
# Definir dados pré-processados globalmente para cada scaler
scalers = {
    'StandardScaler': (StandardScaler(), StandardScaler()),
    'MinMaxScaler': (MinMaxScaler(), MinMaxScaler()),
    'RobustScaler': (RobustScaler(), RobustScaler())
}

print("Pré-processando dados com diferentes scalers...")
# Pré-processar dados para cada tipo de scaler
preprocessed_data = {}
for scaler_name, (X_scaler, y_scaler) in scalers.items():
    print(f"Aplicando {scaler_name}...")
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
print("Pré-processamento concluído!")

# %%
def create_sequences_vectorized(X, y, window_size):
    """Versão vetorizada e otimizada da função create_sequences"""
    print(f"Criando sequências com window_size={window_size}, shape de entrada X={X.shape}, y={y.shape}")

    n_samples = len(X) - window_size

    # Verificação de segurança: garantir que temos amostras suficientes
    if n_samples <= 0:
        print(f"AVISO: window_size {window_size} é maior que o tamanho dos dados {len(X)}!")
        return np.array([]), np.array([])

    X_seq = np.zeros((n_samples, window_size, X.shape[1]))

    for i in range(window_size):
        X_seq[:, i, :] = X[i:i+n_samples]

    y_seq = y[window_size:]

    print(f"Sequências criadas com sucesso: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    return X_seq, y_seq

# %%
# Limite o tamanho do cache para evitar excesso de memória
MAX_CACHE_SIZE = 10
sequence_cache = {}

def get_sequences(scaler_name, window_size):
    """Recupera ou cria sequências com cache limitado"""
    cache_key = f"{scaler_name}_{window_size}"

    if cache_key in sequence_cache:
        print(f"Usando sequências em cache para {cache_key}")
        return sequence_cache[cache_key]

    # Limitar o tamanho do cache
    if len(sequence_cache) >= MAX_CACHE_SIZE:
        # Remover o primeiro item (mais antigo) se o cache estiver cheio
        oldest_key = next(iter(sequence_cache))
        print(f"Cache cheio, removendo item mais antigo: {oldest_key}")
        del sequence_cache[oldest_key]
        # Forçar coleta de lixo para liberar memória
        gc.collect()

    print(f"Criando novas sequências para {scaler_name} com window_size={window_size}")
    data = preprocessed_data[scaler_name]
    X_tr, y_tr = create_sequences_vectorized(data['X_train_s'], data['y_train_s'], window_size)
    X_va, y_va = create_sequences_vectorized(data['X_val_s'], data['y_val_s'], window_size)

    if len(X_tr) == 0 or len(X_va) == 0:
        print(f"ERRO: Sequências vazias geradas para window_size={window_size}!")
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
            print(f"EarlyStopping: Valor NaN ou Inf detectado: {val_loss}. Parando imediatamente.")
            self.early_stop = True
            return self.early_stop

        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"EarlyStopping: Paciência esgotada após {self.patience} épocas sem melhoria.")
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

        print(f"Trial {trial.number}: Iniciando com window_size={window_size}, hidden_size={hidden_size}")

        # Recuperar sequências pré-processadas
        X_tr, y_tr, X_va, y_va = get_sequences(scaler_name, window_size)

        if X_tr is None or len(X_tr) < 10:  # Verificação adicional para garantir dados suficientes
            print(f"Trial {trial.number}: Window size {window_size} muito grande para o conjunto de dados.")
            raise optuna.exceptions.TrialPruned("Window size too large for data split.")

        print(f"Trial {trial.number}: Dados carregados com sucesso. Shapes - X_tr: {X_tr.shape}, y_tr: {y_tr.shape}")

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

        print(f"Trial {trial.number}: Criando modelo com input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")

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

        print(f"Trial {trial.number}: Iniciando treinamento com {max_epochs} épocas máximas")

        for epoch in range(1, max_epochs + 1):
            # Verificar timeout
            if time.time() - start_time > MAX_TRIAL_TIME:
                print(f"Trial {trial.number}: Excedeu o tempo limite de {MAX_TRIAL_TIME}s. Interrompendo.")
                raise optuna.exceptions.TrialPruned("Timeout excedido")

            # Treinamento
            model.train()
            total_loss = 0

            try:
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)

                    # Verificar se a perda é NaN ou inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Trial {trial.number}: NaN ou Inf detectado em loss. Valores: {loss.item()}. Interrompendo.")
                        raise optuna.exceptions.TrialPruned("NaN ou Inf encontrado")

                    loss.backward()

                    # Adicionar clipping de gradiente para estabilidade
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
            except RuntimeError as e:
                # Captura erros específicos do PyTorch (OOM, etc)
                print(f"Trial {trial.number}, Epoch {epoch}: Erro durante o treinamento: {e}")
                raise optuna.exceptions.TrialPruned(f"Runtime error: {str(e)}")

            avg_loss = total_loss / len(train_loader.dataset)

            # Validação - feita a cada época para maior estabilidade
            model.eval()
            running_val = 0

            try:
                with torch.inference_mode():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        val_loss = criterion(pred, yb).item()

                        # Verificar NaN ou Inf
                        if np.isnan(val_loss) or np.isinf(val_loss):
                            print(f"Trial {trial.number}, Epoch {epoch}: NaN ou Inf na validação: {val_loss}. Interrompendo.")
                            raise optuna.exceptions.TrialPruned("NaN ou Inf encontrado na validação")

                        running_val += val_loss * xb.size(0)
            except RuntimeError as e:
                # Captura erros específicos do PyTorch (OOM, etc)
                print(f"Trial {trial.number}, Epoch {epoch}: Erro durante a validação: {e}")
                raise optuna.exceptions.TrialPruned(f"Runtime error na validação: {str(e)}")

            avg_val = running_val / len(val_loader.dataset)

            print(f"Trial {trial.number}, Epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={avg_val:.6f}")

            if avg_val < best_avg_val_loss:
                best_avg_val_loss = avg_val
                print(f"Trial {trial.number}: Novo melhor val_loss: {best_avg_val_loss:.6f}")

            # Ajustar learning rate baseado na performance
            scheduler.step(avg_val)

            # Checagem de early stopping
            if early_stopping(avg_val):
                print(f"Trial {trial.number}: Early stopping ativado no epoch {epoch}")
                break

            # Reportar ao Optuna para pruning
            trial.report(avg_val, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number}: Pruned pelo Optuna após epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        # Limpar memória
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Trial {trial.number}: Concluído com melhor val_loss = {best_avg_val_loss:.6f}")
        return best_avg_val_loss

    except optuna.exceptions.TrialPruned as e:
        # Este é um tipo esperado de exceção para pruning, simplesmente repassamos
        print(f"Trial {trial.number}: Pruned - {str(e)}")
        raise
    except Exception as e:
        # Capturar qualquer exceção não tratada e imprimir detalhes completos
        print(f"Trial {trial.number}: ERRO CRÍTICO:")
        print(traceback.format_exc())  # Imprime o stack trace completo
        raise optuna.exceptions.TrialPruned(f"Exceção: {str(e)}")

# %%
# Atualize a configuração da otimização para gravar logs mais detalhados
print("Configurando estudo Optuna com log verboso...")

# Configuração dos prunadores e samplers do Optuna para maior eficiência
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=5,
    interval_steps=1
)

study = optuna.create_study(
    direction='minimize',
    pruner=pruner,
    sampler=optuna.samplers.TPESampler(n_startup_trials=5)
)

# Número de trials e configuração de paralelismo
n_trials = 30
n_jobs = 1  # Mantido em 1 para evitar problemas de paralelismo

print(f"Iniciando otimização com {n_trials} trials, {n_jobs} job(s)...")

# %%
# Execute a otimização com timeout global e capture exceções globais
try:
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=3600)  # 1 hora de timeout total
except KeyboardInterrupt:
    print("Otimização interrompida pelo usuário.")
except Exception as e:
    print("Erro global na otimização:")
    print(traceback.format_exc())

# %%
# Imprima os resultados se houver trials bem-sucedidos
if len(study.trials) > 0:
    print("\n" + "="*50)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("="*50)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Trials completados: {len(completed_trials)}/{len(study.trials)}")

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Trials podados: {len(pruned_trials)}/{len(study.trials)}")

    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"Trials com falha: {len(failed_trials)}/{len(study.trials)}")

    if study.best_trial:
        print("\nMelhor trial:")
        trial = study.best_trial
        print(f"  ID: {trial.number}")
        print(f"  Valor (menor val_loss): {trial.value}")
        print("  Parâmetros: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Salvar os melhores parâmetros
        with open('results_optuna.json', "w") as f:
            json.dump(trial.params, f, indent=4)
        print("\nMelhores parâmetros salvos em 'results_optuna.json'")

        # Tentar gerar visualizações
        try:
            import plotly
            print("\nGerando visualizações...")

            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image('optimization_history.png')
            print("- Histórico de otimização salvo em 'optimization_history.png'")

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image('param_importances.png')
            print("- Importância dos parâmetros salva em 'param_importances.png'")


        except Exception as viz_error:
            print(f"\nNão foi possível gerar visualizações: {viz_error}")
            print("Verifique se plotly está instalado corretamente.")
    else:
        print("\nNenhum trial foi concluído com sucesso.")
else:
    print("\nNenhum trial foi executado.")
