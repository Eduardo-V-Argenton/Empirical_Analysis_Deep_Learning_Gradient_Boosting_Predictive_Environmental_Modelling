import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# 1) Pasta raiz
base_dir = '/home/eduardo/Documentos/Water-Cycle-Neural-Network/results/models_test/11'

# 2) Carrega todos os result.json
data = {}
for root, dirs, files in os.walk(base_dir):
    if 'result.json' in files:
        key = os.path.relpath(root, base_dir)
        with open(os.path.join(root, 'result.json'), 'r') as f:
            obj = json.load(f)
        df = pd.DataFrame(obj.get('metrics_per_feature', []))
        if not df.empty:
            df = df.set_index('feature').astype(float)
            data[key] = df

if not data:
    print(f'Erro: nenhum result.json encontrado em {base_dir}', file=sys.stderr)
    sys.exit(1)

# 3) Concatena
all_df = pd.concat(data, axis=1)

# 4) Métricas e features
metrics = ['MSE', 'RMSE', 'R2', 'NRMSE', 'std']
features = all_df.index
n_files = len(data)
x = np.arange(len(features))
width = 0.8 / n_files

# 5) Cria figura
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), sharex=True)

for i, metr in enumerate(metrics):
    ax = axes[i]
    # desenha cada método
    for j, key in enumerate(data):
        vals = all_df[(key, metr)].values
        ax.bar(x + j*width, vals, width=width, label=key)
    # se a variação é muito grande, usa log
    vals_all = all_df.xs(metr, axis=1, level=1).values.flatten()
    if vals_all.max() / (vals_all.min()+1e-12) > 50 and metr in ['MSE', 'RMSE', 'std']:
        ax.set_yscale('log')
        ax.set_ylabel(f'{metr} (log scale)')
    else:
        ax.set_ylabel(metr)
        if metr == 'R2':
            ax.set_ylim(0, 1.05)
    ax.set_title(f'{metr} por feature')
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 6) Ajustes finais e save
plt.xticks(x + width*(n_files-1)/2, features, rotation=45, ha='right')
plt.tight_layout()
out_path = os.path.join(base_dir, 'comparison_metrics.png')
plt.savefig(out_path, dpi=300)
print(f'Gráfico salvo em {out_path}')
