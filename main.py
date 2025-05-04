# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme,facet_wrap,scale_color_manual,theme_bw
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump
from basic import Model, features_y, features_X

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
df = pd.read_csv('data/cleaned_data.csv')
df.head()
# %%

# %%
# Features que serão usadas
X = df[features_X]
y = df[features_y]
# %%

def create_sequences(Xs, ys, window_size):
    X_seq, y_seq = [], []
    for i in range(len(Xs) - window_size):
        X_seq.append(Xs[i:i+window_size])
        y_seq.append(ys[i+window_size])
    return np.array(X_seq), np.array(y_seq)

# %%
all_metrics = []
all_avg_losses = []
all_df_plots = []
best_model_state_dict = None
best_nrmse = float('inf')

# Hiperparâmetros
input_size  = 0
output_size  = 0
hidden_size = 128
num_layers  = 4
lr          = 0.00017085561327576825
batch_size  = 8
dropout     = 0.4
weight_decay = 2.1768208421129883e-06
epochs      = 100
window_size = 6
bidirectional = True

# %%
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

    print(f"Fold {fold+1}")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Escalonamento novo para cada fold
    X_scaler = MinMaxScaler().fit(X_train_fold)
    y_scaler = MinMaxScaler().fit(y_train_fold)

    X_train_s = X_scaler.transform(X_train_fold)
    X_val_s   = X_scaler.transform(X_val_fold)
    y_train_s = y_scaler.transform(y_train_fold)
    y_val_s   = y_scaler.transform(y_val_fold)


    # Criar sequências para LSTM
    X_tr, y_tr = create_sequences(X_train_s, y_train_s, window_size)
    X_va, y_va = create_sequences(X_val_s,   y_val_s,   window_size)

    input_size = X_tr.shape[2]
    output_size = y_tr.shape[1]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                            torch.from_numpy(y_tr).float()),
                            batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_va).float(),
                                            torch.from_numpy(y_va).float()),
                            batch_size, shuffle=False)

    # Modelo

    model = Model(input_size, hidden_size, num_layers, output_size, dropout, bidirectional).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

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
        all_avg_losses.append(avg_loss)

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

    X_test_t = torch.from_numpy(X_va).float().to(device)
    y_test_t = torch.from_numpy(y_va).float().to(device)

    model.eval()
    with torch.inference_mode():
        y_pred_t = model(X_test_t)    # shape (N, n_features)

    y_pred = y_scaler.inverse_transform(y_pred_t.cpu().numpy())
    y_true = y_scaler.inverse_transform(y_test_t.cpu().numpy())

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

    metrics_per_feat = pd.DataFrame({
        'feature': features_y,
        'std':     std,
        'MSE':     mse,
        'RMSE':    rmse,
        'NRMSE':   nrmse,
        'R2':      r2
    })

    summed_nrmse =  sum(metrics_per_feat['NRMSE'].values)
    if best_nrmse > summed_nrmse:
        best_nrmse = summed_nrmse
        best_model_state_dict = model.state_dict()
    print(metrics_per_feat)
    all_metrics.append(metrics_per_feat)

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

    all_df_plots.append(df_plot)


# %%
df_all = pd.concat(all_metrics, ignore_index=True)

# 2) Média por feature
overall_per_feature = df_all.groupby('feature').mean()

print("Métricas médias por feature:")
print(overall_per_feature)

# 3) Se quiser também a média global (todas as features juntas)
overall_global = df_all.drop(columns='feature').mean().to_frame().T
overall_global.index = ['overall']

print("\nMétrica global (todas as features):")
print(overall_global)

# %%
df_plot = pd.concat(all_df_plots, ignore_index=True)

# Agora calcula a média agrupando por 'index', 'type' e 'feature'
df_plot_mean = (
    df_plot
    .groupby(['index', 'type', 'feature'], as_index=False)
    .mean()
)
# Junta todos os DataFrames em um só
plot = (
    ggplot(df_plot_mean, aes(x='index', y='value', color='type'))
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
    "features_X":          list(features_X),
    "features_y":          features_y,
    "window_size":         window_size,
    "input_size":          input_size,
    "output_size":         output_size,
    "hidden_size":         hidden_size,
    "num_layers":          num_layers,
    "learning_rate":       lr,
    "batch_size":          batch_size,
    "epochs":              epochs,
    "metrics_per_feature": overall_per_feature.to_dict(orient='records'),
    "metrics_total":       overall_global.to_dict(orient='records'),
    "train_losses":        all_avg_losses,
}

filename = f"results/main/{next_number}/result.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=4)

print("📊 Todos os resultados foram gravados em results.json")

# %%
filename = f"results/main/{next_number}/model.h5"
torch.save(best_model_state_dict, filename)

# %%
X_scaler_final = MinMaxScaler().fit(X)
y_scaler_final = MinMaxScaler().fit(y)
dump(X_scaler_final, f"results/main/{next_number}/x_scaler.pkl")
dump(y_scaler_final, f"results/main/{next_number}/y_scaler.pkl")
