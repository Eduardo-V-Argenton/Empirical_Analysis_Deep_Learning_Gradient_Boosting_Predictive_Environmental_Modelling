# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme, facet_wrap, scale_color_manual, theme_bw
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump
from basic import Model, features_y, features_X
import matplotlib.pyplot as plt

# %%
# Set device and ensure reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
    print(f"HIP version: {torch.version.hip}")

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# %%
# Load data
df = pd.read_csv('data/cleaned_data.csv')
print(f"Data loaded with shape: {df.shape}")
print(df.head())

# %%
# Extract features
X = df[features_X]
y = df[features_y]
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Features X: {features_X}")
print(f"Features y: {features_y}")

# %%
def create_sequences(Xs, ys, window_size):
    """Create sequences for LSTM input from time series data."""
    X_seq, y_seq = [], []
    for i in range(len(Xs) - window_size):
        X_seq.append(Xs[i:i+window_size])
        y_seq.append(ys[i+window_size])
    return np.array(X_seq), np.array(y_seq)

# %%
# Initialize variables to track the best model
best_metrics = None
best_losses = None
best_val_losses = None  # Added to track validation losses
best_df_plots = None
best_model_state_dict = None
best_nrmse = float('inf')
best_fold = -1  # Track the best fold

# Hyperparameters
input_size = len(features_X)  # Initialize with correct dimensions
output_size = len(features_y)
hidden_size = 128
num_layers = 2
lr = 0.0003177200256878974
batch_size = 32
dropout = 0.4
weight_decay = 1.9543450419518126e-05
epochs = 100
window_size = 12
bidirectional = False
patience = 20  # Early stopping patience

# Create results directory if it doesn't exist
os.makedirs("results/main", exist_ok=True)

# %%
# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []  # Store metrics for each fold

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n{'='*50}\nFold {fold+1}\n{'='*50}")

    # Split data for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Train set: {X_train_fold.shape[0]} samples")
    print(f"Validation set: {X_val_fold.shape[0]} samples")

    # Scale features for this fold
    X_scaler = MinMaxScaler().fit(X_train_fold)
    y_scaler = MinMaxScaler().fit(y_train_fold)

    X_train_s = X_scaler.transform(X_train_fold)
    X_val_s = X_scaler.transform(X_val_fold)
    y_train_s = y_scaler.transform(y_train_fold)
    y_val_s = y_scaler.transform(y_val_fold)

    # Create sequences for LSTM
    X_tr, y_tr = create_sequences(X_train_s, y_train_s, window_size)
    X_va, y_va = create_sequences(X_val_s, y_val_s, window_size)

    input_size = X_tr.shape[2]
    output_size = y_tr.shape[1]
    print(f"Sequence input size: {input_size}")
    print(f"Sequence output size: {output_size}")

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()),
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    model = Model(input_size, hidden_size, num_layers, output_size, dropout, bidirectional).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # Track losses and predictions
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_since_best = 0
    best_epoch = 0
    fold_best_model = None

    # Training loop
    for epoch in range(1, epochs+1):
        # Training phase
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # Learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Check for improvement and implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_since_best = 0
            # Save the best model for this fold
            fold_best_model = model.state_dict().copy()
        else:
            epochs_since_best += 1

        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train loss: {avg_loss:.5f} | Val loss: {avg_val_loss:.5f} | LR: {current_lr:.1e} | Best: {best_epoch}")

        # Early stopping
        if epochs_since_best > patience:
            print(f'Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss: {best_val_loss:.5f}')
            break

    # Load the best model for this fold for evaluation
    model.load_state_dict(fold_best_model)

    # Evaluate on validation set
    X_test_t = torch.from_numpy(X_va).float().to(device)
    y_test_t = torch.from_numpy(y_va).float().to(device)

    model.eval()
    with torch.inference_mode():
        y_pred_t = model(X_test_t)

    # Transform predictions back to original scale
    y_pred = y_scaler.inverse_transform(y_pred_t.cpu().numpy())
    y_true = y_scaler.inverse_transform(y_test_t.cpu().numpy())

    # Calculate metrics
    mse = ((y_true - y_pred)**2).mean(axis=0)
    rmse = np.sqrt(mse)
    std = y_true.std(axis=0)
    nrmse = rmse / std
    r2 = [r2_score(y_true[:,i], y_pred[:,i]) for i in range(len(features_y))]

    metrics_per_feat = pd.DataFrame({
        'feature': features_y,
        'std': std,
        'MSE': mse,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'R2': r2
    })

    print("\nMetrics for this fold:")
    print(metrics_per_feat)

    # Store fold metrics
    fold_metrics.append({
        'fold': fold + 1,
        'metrics': metrics_per_feat,
        'avg_nrmse': np.mean(nrmse)
    })

    # Prepare visualization data
    train_val_split = len(train_idx)
    start = window_size  # Start from first prediction point
    end = start + len(y_true)

    if isinstance(df.index, pd.DatetimeIndex):
        idx_test = df.index[val_idx[start:end]]
    else:
        idx_test = np.arange(val_idx[0] + start, val_idx[0] + end)

    # Create DataFrames for plotting
    df_true = pd.DataFrame(y_true, index=idx_test, columns=features_y)
    df_pred = pd.DataFrame(y_pred, index=idx_test, columns=features_y)

    # Select a slice for visualization
    max_points = min(200, len(df_true))
    slice_true = df_true.iloc[0:max_points]
    slice_pred = df_pred.iloc[0:max_points]

    # Prepare data for plotting
    df_plot = (
        pd.concat([
            slice_true.assign(type="Real"),
            slice_pred.assign(type="Predicted")
        ])
        .reset_index()
        .melt(id_vars=['index', 'type'], value_vars=features_y,
              var_name='feature', value_name='value')
    )

    # Check if this fold has better performance
    summed_nrmse = sum(metrics_per_feat['NRMSE'].values)
    if best_nrmse > summed_nrmse:
        best_nrmse = summed_nrmse
        best_metrics = metrics_per_feat.copy()
        best_model_state_dict = fold_best_model.copy()  # Use the best model from this fold
        best_df_plots = df_plot.copy()
        best_losses = train_losses.copy()
        best_val_losses = val_losses.copy()  # Save validation losses too
        best_fold = fold + 1

# Print best fold results
print(f"\n{'='*50}")
print(f"Best results from fold {best_fold}")
print(f"{'='*50}")
print(best_metrics)

# %%
# Create results directory structure
def get_next_number(folder):
    """Get next available number for results directory."""
    try:
        # Check if directory exists and has contents
        if os.path.exists(folder) and os.listdir(folder):
            numbers = [int(d) for d in os.listdir(folder) if d.isdigit()]
            next_number = max(numbers) + 1 if numbers else 1
        else:
            next_number = 1

        # Create the directory
        os.makedirs(f"{folder}/{next_number}", exist_ok=True)
        return next_number
    except Exception as e:
        print(f"Error creating directory: {e}")
        # Fallback to timestamp
        import time
        return int(time.time())

# Get directory for results
next_number = get_next_number("results/main")
result_dir = f"results/main/{next_number}"
print(f"Saving results to {result_dir}")

# %%
# Plot and save results
df_plot = best_df_plots

# Create plot
plot = (
    ggplot(df_plot, aes(x='index', y='value', color='type'))
    + geom_line()
    + facet_wrap('~feature', scales='free_y', ncol=1)
    + labs(
        title="Real vs Predicted (test slice)",
        x="Timestamp (or sample)",
        y="Value",
        color=""
    )
    + theme_bw()
    + theme(figure_size=(12, 16))
)

# Save plot
plot_filename = f"{result_dir}/comparison.png"
plot.save(plot_filename, dpi=300)
print(f"Plot saved to {plot_filename}")

# %%
# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(best_losses, label='Training Loss')
plt.plot(best_val_losses, label='Validation Loss')
plt.title(f'Training and Validation Loss - Fold {best_fold}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{result_dir}/losses.png", dpi=300)

# %%
# Save results to JSON
results = {
    "best_fold": best_fold,
    "features_X": list(features_X),
    "features_y": features_y,
    "window_size": window_size,
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "bidirectional": bidirectional,
    "dropout": dropout,
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs,
    "metrics_per_feature": best_metrics.to_dict(orient='records'),
    "train_losses": best_losses,
    "val_losses": best_val_losses,
}

json_filename = f"{result_dir}/result.json"
with open(json_filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"📊 Results saved to {json_filename}")

# %%
# Save model state
model_filename = f"{result_dir}/model.pt"  # Using .pt extension which is more standard
torch.save(best_model_state_dict, model_filename)
print(f"🧠 Model saved to {model_filename}")

# %%
# Save scalers using the full dataset for inference
X_scaler_final = MinMaxScaler().fit(X)
y_scaler_final = MinMaxScaler().fit(y)
dump(X_scaler_final, f"{result_dir}/x_scaler.pkl")
dump(y_scaler_final, f"{result_dir}/y_scaler.pkl")
print("📏 Scalers saved")

# %%
# Save a summary of fold performances
fold_summary = pd.DataFrame([
    {
        'fold': fm['fold'],
        'avg_nrmse': fm['avg_nrmse'],
        'is_best': fm['fold'] == best_fold
    } for fm in fold_metrics
])

fold_summary.to_csv(f"{result_dir}/fold_summary.csv", index=False)
print("📋 Fold summary saved")
