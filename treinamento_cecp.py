
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
# carregar dados
data_dir = 'dataset/'  

# Função para computar stats por ciclo
def compute_stats(df):
    return pd.DataFrame({
        'mean': df.mean(axis=1),
        'std': df.std(axis=1),
        'min': df.min(axis=1),
        'max': df.max(axis=1)
    })

# Carregar todos os sensores relevantes
ps1 = pd.read_csv(data_dir + 'PS1.txt', sep='\t', header=None)
ps2 = pd.read_csv(data_dir + 'PS2.txt', sep='\t', header=None)
ps3 = pd.read_csv(data_dir + 'PS3.txt', sep='\t', header=None)
ps4 = pd.read_csv(data_dir + 'PS4.txt', sep='\t', header=None)
ps5 = pd.read_csv(data_dir + 'PS5.txt', sep='\t', header=None)
ps6 = pd.read_csv(data_dir + 'PS6.txt', sep='\t', header=None)

eps1 = pd.read_csv(data_dir + 'EPS1.txt', sep='\t', header=None)

fs1 = pd.read_csv(data_dir + 'FS1.txt', sep='\t', header=None)
fs2 = pd.read_csv(data_dir + 'FS2.txt', sep='\t', header=None)

ts1 = pd.read_csv(data_dir + 'TS1.txt', sep='\t', header=None)
ts2 = pd.read_csv(data_dir + 'TS2.txt', sep='\t', header=None)
ts3 = pd.read_csv(data_dir + 'TS3.txt', sep='\t', header=None)
ts4 = pd.read_csv(data_dir + 'TS4.txt', sep='\t', header=None)

vs1 = pd.read_csv(data_dir + 'VS1.txt', sep='\t', header=None)

ce = pd.read_csv(data_dir + 'CE.txt', sep='\t', header=None)
cp = pd.read_csv(data_dir + 'CP.txt', sep='\t', header=None)

# Target: média de SE por ciclo
se = pd.read_csv(data_dir + 'SE.txt', sep='\t', header=None)
y = se.mean(axis=1).values

# Carregar profile.txt para targets adicionais
profile = pd.read_csv(data_dir + 'profile.txt', sep='\t', header=None)
profile.columns = ['Cooler_%', 'Valve_%', 'Pump_Leakage', 'Accumulator_bar', 'Stable_Flag']

# Computar stats para todas as features
ps1_stats = compute_stats(ps1).add_prefix('PS1_')
ps2_stats = compute_stats(ps2).add_prefix('PS2_')
ps3_stats = compute_stats(ps3).add_prefix('PS3_')
ps4_stats = compute_stats(ps4).add_prefix('PS4_')
ps5_stats = compute_stats(ps5).add_prefix('PS5_')
ps6_stats = compute_stats(ps6).add_prefix('PS6_')

eps1_stats = compute_stats(eps1).add_prefix('EPS1_')

fs1_stats = compute_stats(fs1).add_prefix('FS1_')
fs2_stats = compute_stats(fs2).add_prefix('FS2_')

ts1_stats = compute_stats(ts1).add_prefix('TS1_')
ts2_stats = compute_stats(ts2).add_prefix('TS2_')
ts3_stats = compute_stats(ts3).add_prefix('TS3_')
ts4_stats = compute_stats(ts4).add_prefix('TS4_')

vs1_stats = compute_stats(vs1).add_prefix('VS1_')

ce_stats = compute_stats(ce).add_prefix('CE_')
cp_stats = compute_stats(cp).add_prefix('CP_')

# Combinar todas as features em X_df
X_df = pd.concat([ps1_stats, ps2_stats, ps3_stats, ps4_stats, ps5_stats, ps6_stats,
                  eps1_stats, fs1_stats, fs2_stats, ts1_stats, ts2_stats, ts3_stats, ts4_stats,
                  vs1_stats, ce_stats, cp_stats], axis=1)

# Adicionar target e profile para avaliação
eval_df = X_df.copy()
eval_df['Efficiency'] = y
eval_df = pd.concat([eval_df, profile], axis=1)

# Avaliação da Base de Dados
print("Cabeçalho (primeiras 5 linhas, features agregadas + Efficiency + Profile):")
print(eval_df.head())
print("\nDescrição Estatística (todas features + targets):")
print(eval_df.describe())
print("\nCorrelações (Pearson, focando em Efficiency):")
print(eval_df.corr()['Efficiency'].sort_values(ascending=False))  # Correlações com Efficiency

# Preparo para ML: X normalizado, split train/test
X_raw = X_df.values
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min + 1e-8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class HydraulicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HydraulicDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)

def train_model(model, dataloader, X_test_tensor, y_test, epochs=100, lr=0.01, l1_alpha=0.0, l2_alpha=0.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_alpha)
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = mse_loss(outputs, batch_y)
            if l1_alpha > 0:
                l1_reg = sum([torch.norm(p, 1) for p in model.parameters() if p.dim() > 0])
                loss += l1_alpha * l1_reg
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        preds_test = model(X_test_tensor).numpy().flatten()
        mse = np.mean((preds_test - y_test)**2)
    return mse, preds_test

# Treinar modelos
input_size = X.shape[1]
model_linear = LinearModel(input_size)
mse_linear, preds_linear = train_model(model_linear, train_dataloader, X_test_tensor, y_test, l1_alpha=0, l2_alpha=0)

model_ridge = LinearModel(input_size)
mse_ridge, preds_ridge = train_model(model_ridge, train_dataloader, X_test_tensor, y_test, l1_alpha=0, l2_alpha=0.01)

model_lasso = LinearModel(input_size)
mse_lasso, preds_lasso = train_model(model_lasso, train_dataloader, X_test_tensor, y_test, l1_alpha=0.01, l2_alpha=0)

# Comparação
print(f"\nResultados da Comparação (MSE no Test Set):")
print(f"Linear MSE: {mse_linear:.4f}")
print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Lasso MSE: {mse_lasso:.4f}")
best_mse = min(mse_linear, mse_ridge, mse_lasso)
if best_mse == mse_linear:
    best_preds = preds_linear
    best_model = model_linear
    print("Melhor modelo: Linear")
elif best_mse == mse_ridge:
    best_preds = preds_ridge
    best_model = model_ridge
    print("Melhor modelo: Ridge")
else:
    best_preds = preds_lasso
    best_model = model_lasso
    print("Melhor modelo: Lasso")

# Gráficos de comparação
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(y_test, preds_linear); axs[0].plot([0,100], [0,100], 'r--'); axs[0].set_title('Linear')
axs[1].scatter(y_test, preds_ridge); axs[1].plot([0,100], [0,100], 'r--'); axs[1].set_title('Ridge')
axs[2].scatter(y_test, preds_lasso); axs[2].plot([0,100], [0,100], 'r--'); axs[2].set_title('Lasso')
plt.savefig('comparison.png')
plt.show()

# Preparar dados do test set para 3D
test_indices = np.arange(len(y))[-len(y_test):]  
test_df = X_df.iloc[test_indices]

# 3D para o melhor modelo
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_df['PS1_mean'], test_df['FS1_mean'], y_test, c='b', marker='o', label='Atual')
ax.scatter(test_df['PS1_mean'], test_df['FS1_mean'], best_preds, c='r', marker='^', label='Predito')
ax.set_xlabel('Pressão Média (PS1, bar)')
ax.set_ylabel('Fluxo Médio (FS1, l/min)')
ax.set_zlabel('Eficiência (%)')
ax.set_facecolor('white')
ax.grid(True)
ax.legend()
ax.set_title('Melhor Modelo')
plt.savefig('3d_plot.png')
plt.show()

# Salva melhor modelo
torch.save(best_model.state_dict(), 'melhor_modelo.pth')
np.save('X_min.npy', X_min)
np.save('X_max.npy', X_max)
print("\nModelo salvo como 'melhor_modelo.pth'")
print("Parâmetros de normalização salvos como 'X_min.npy' e 'X_max.npy'")

import json

# Exemplo para o modelo completo (ajuste nomes se for o físico)
results = {
    "model_type": "Completo",  
    "mse_linear": mse_linear,
    "mse_ridge": mse_ridge,
    "mse_lasso": mse_lasso,
    "best_model": "Linear" if best_mse == mse_linear else "Ridge" if best_mse == mse_ridge else "Lasso",
    "best_mse": best_mse,
    "num_cycles": len(y),
    "num_features": X.shape[1],
    "split": "80% treino / 20% teste",
    "epochs": 100,
    "optimizer": "Adam",
    "lr": 0.01,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Salvar (use nomes diferentes para cada treinamento)
with open('training_results_completo.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Resultados salvos em 'training_results_completo.json'")