import pandas as pd

# dataset
data_dir = 'dataset/'  

# Função para computar stats por ciclo
def compute_stats(df):
    return pd.DataFrame({
        'mean': df.mean(axis=1),
        'std': df.std(axis=1),
        'min': df.min(axis=1),
        'max': df.max(axis=1)
    })

# Carregar todos os sensores
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

# Target
se = pd.read_csv(data_dir + 'SE.txt', sep='\t', header=None)
y = se.mean(axis=1).values

# Profile para contexto
profile = pd.read_csv(data_dir + 'profile.txt', sep='\t', header=None)
profile.columns = ['Cooler_%', 'Valve_%', 'Pump_Leakage', 'Accumulator_bar', 'Stable_Flag']

# Stats de todos os sensores
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

# Combinar tudo
X_df = pd.concat([ps1_stats, ps2_stats, ps3_stats, ps4_stats, ps5_stats, ps6_stats,
                  eps1_stats, fs1_stats, fs2_stats, ts1_stats, ts2_stats, ts3_stats, ts4_stats,
                  vs1_stats, ce_stats, cp_stats], axis=1)

eval_df = X_df.copy()
eval_df['Efficiency'] = y
eval_df = pd.concat([eval_df, profile], axis=1)

# Avaliação completa
print("=== Cabeçalho (primeiras 10 linhas) ===")
print(eval_df.head(10))

print("\n=== Descrição Estatística Completa ===")
print(eval_df.describe())

print("\n=== Correlações com Efficiency ===")
print(eval_df.corr()['Efficiency'].sort_values(ascending=False))

print("\n=== Informações Gerais do Dataset ===")
print(f"Número total de ciclos: {len(eval_df)}")
print(f"Número de features agregadas: {len(X_df.columns)}")
print(f"Valores únicos em cada coluna do profile:")
for col in profile.columns:
    print(f"  {col}: {sorted(eval_df[col].unique())}")