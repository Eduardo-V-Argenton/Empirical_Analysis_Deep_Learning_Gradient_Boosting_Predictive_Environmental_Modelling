# %%
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import plotly
import json

# %%
df = pd.read_csv('data/cleaned_data.csv')

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_sequences(Xs, ys, window_size):
    X_seq, y_seq = [], []
    for i in range(len(Xs) - window_size):
        X_seq.append(Xs[i:i+window_size])
        y_seq.append(ys[i+window_size])
    return np.array(X_seq), np.array(y_seq)

# %%
def objective(trial):
    # --- 1. Sugerir Hiperparâmetros ---
    # Define os hiperparâmetros a serem otimizados e seus intervalos/opções
    window_size = trial.suggest_int('window_size', 6, 48, step=6) # Exemplo: se window_size for tunado
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5, step=0.1) # Taxa de dropout a ser usada
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    bidirectional_lstm = trial.suggest_categorical('bidirectional', [True, False])
    loss_name = trial.suggest_categorical('loss', ['mse', 'mae', 'hubber'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    scaler_name = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])

    features_y = ['Temperature', 'Precipitation_log', 'Humidity', 'Wind_Speed_kmh',
                'Soil_Moisture', 'Soil_Temperature',
                'Wind_Dir_Sin', 'Wind_Dir_Cos']

    features_X = df.columns.drop(['Timestamp', 'Wind_Direction', 'Precipitation'])
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
    if scaler_name == 'StandardScaler':
        X_scaler = StandardScaler();
        y_scaler = StandardScaler();

    elif scaler_name == 'MinMaxScaler':
        X_scaler = MinMaxScaler();
        y_scaler = MinMaxScaler();

    else:
        X_scaler = RobustScaler();
        y_scaler = RobustScaler();

    X_scaler.fit(X_train)
    y_scaler.fit(y_train)

    X_train_s = X_scaler.transform(X_train)
    X_val_s   = X_scaler.transform(X_val)
    X_test_s  = X_scaler.transform(X_test)

    y_train_s = y_scaler.transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    y_test_s  = y_scaler.transform(y_test)

    # Recalcular X_tr, y_tr, X_va, y_va com o window_size sugerido
    # Lembre-se que X_train_s, y_train_s, etc. foram calculados fora
    X_tr, y_tr = create_sequences(X_train_s, y_train_s, window_size)
    X_va, y_va = create_sequences(X_val_s,  y_val_s,  window_size)
    # Cuidado: Se window_size mudar, o número de amostras muda!

    # --- 2. Preparar Dados e Modelo com os Hiperparâmetros Sugeridos ---
    if len(X_tr) == 0 or len(X_va) == 0: # Evita erro se window_size for muito grande
         raise optuna.exceptions.TrialPruned("Window size too large for data split.")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
                              batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()),
                            batch_size, shuffle=False)

    input_size = X_tr.shape[2]
    output_size = y_tr.shape[1]

    class GRU(nn.Module):
        def __init__(self, in_size, hid_size, n_layers, out_size, dropout_p, bidirectional):
            super().__init__()
            self.num_directions = 2 if bidirectional else 1
            self.gru = nn.GRU(
                in_size,
                hid_size,
                n_layers,
                batch_first=True,
                dropout=dropout_p if n_layers > 1 else 0,  # Dropout só entre camadas se n_layers > 1
                bidirectional=bidirectional
            )
            # Camada final
            self.fc = nn.Linear(hid_size * self.num_directions, out_size)
            # Dropout externo (opcional)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, x):
            # x: (B, T, in_size)
            gru_out, h_n = self.gru(x)
            # h_n: (num_layers * num_directions, B, hid_size)
            if self.num_directions == 2:
                # Última camada forward + backward
                fwd = h_n[-2]  # última camada, direção forward
                bwd = h_n[-1]  # última camada, direção backward
                h_cat = torch.cat((fwd, bwd), dim=1)
            else:
                # Só pegar o último hidden state da última camada
                h_cat = h_n[-1]

            # Dropout antes da FC (se desejado)
            h_cat = self.dropout(h_cat)
            return self.fc(h_cat)


    model = GRU(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional_lstm).to(device)
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    if loss_name == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss()

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- 3. Loop de Treino e Validação ---
    epochs = 50 # Use um número menor de épocas para cada trial, ou use early stopping
    best_avg_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
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

        # --- Logging ---
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss: {avg_loss:.5f}") # Adicionado LR

        # --- Validação ---
        model.eval()
        running_val = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                running_val += criterion(pred, yb).item() * xb.size(0)
        avg_val = running_val / len(val_loader.dataset)

        if avg_val < best_avg_val_loss:
            best_avg_val_loss = avg_val

        # --- (Opcional, mas recomendado) Pruning ---
        # Reporta o resultado intermediário para o Optuna
        trial.report(avg_val, epoch)
        # Verifica se o Optuna acha que este trial deve ser interrompido
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # --- 4. Retornar a Métrica a ser Otimizada ---
    # Queremos minimizar a melhor perda de validação encontrada neste trial
    return best_avg_val_loss

# --- Executar o Estudo de Otimização ---
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
# direction='minimize' porque queremos minimizar a loss
# pruner=... é para ativar o pruning (parar trials ruins cedo)
# n_trials é o número de combinações de hiperparâmetros a testar
study.optimize(objective, n_trials=50) # Ajuste n_trials conforme seu tempo/recursos

# --- Ver os Resultados ---
print("Melhor trial:")
trial = study.best_trial
print(f"  Valor (menor val_loss): {trial.value}")
print("  Parâmetros: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# %%
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('optimization_history.png')
fig = optuna.visualization.plot_param_importances(study)
fig.write_image('param_importances.png')

with open('results_optuna.json', "w") as f:
    json.dump(trial.params, f, indent=4)
