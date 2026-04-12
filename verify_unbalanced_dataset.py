import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMBALANCE_RATIO_THRESHOLD = 1.5  # Si majority/minority > umbral → desbalanceado
REGISTRY_PATH = 'datasets_registry.json'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open(REGISTRY_PATH, 'r') as f:
    datasets_registry = json.load(f)

list_datasets = list(datasets_registry.keys())

# Selección del dataset: argumento posicional o interactivo
if len(sys.argv) > 1 and sys.argv[1] in datasets_registry:
    dataset_name = sys.argv[1]
elif len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) < len(list_datasets):
    dataset_name = list_datasets[int(sys.argv[1])]
else:
    print("Datasets disponibles:")
    for i, name in enumerate(list_datasets):
        print(f"  [{i}] {name}")
    choice = input("Selecciona dataset (nombre o índice): ").strip()
    if choice in datasets_registry:
        dataset_name = choice
    elif choice.isdigit() and int(choice) < len(list_datasets):
        dataset_name = list_datasets[int(choice)]
    else:
        print(f"Opción '{choice}' no válida.")
        sys.exit(1)

config = datasets_registry[dataset_name]
csv_path = config['csv']
target_column = config.get('target_column', 'target')
sep = config.get('sep', ',')

df = pd.read_csv(csv_path, sep=sep)

# --- Análisis numérico ---
counts = df[target_column].value_counts()
percentages = df[target_column].value_counts(normalize=True) * 100
majority = counts.iloc[0]
minority = counts.iloc[-1]
imbalance_ratio = majority / minority

summary = pd.DataFrame({
    'Conteo': counts,
    'Porcentaje (%)': percentages.round(2)
})

print("=" * 50)
print(f"  Dataset        : {dataset_name}  ({csv_path})")
print(f"  Columna target : '{target_column}'")
print(f"  Total muestras : {len(df)}")
print("=" * 50)
print(summary.to_string())
print("-" * 50)
print(f"Clase mayoritaria : {counts.index[0]}  ({majority} muestras, {percentages.iloc[0]:.2f}%)")
print(f"Clase minoritaria : {counts.index[-1]}  ({minority} muestras, {percentages.iloc[-1]:.2f}%)")
print(f"Ratio de desbalance (mayor/menor): {imbalance_ratio:.1f}x")

if imbalance_ratio > IMBALANCE_RATIO_THRESHOLD:
    print(f"\n>>> DATASET DESBALANCEADO (ratio {imbalance_ratio:.1f} > umbral {IMBALANCE_RATIO_THRESHOLD})")
else:
    print(f"\n>>> Dataset BALANCEADO (ratio {imbalance_ratio:.1f} <= umbral {IMBALANCE_RATIO_THRESHOLD})")
print("=" * 50)

# --- Visualización ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=target_column, data=df, ax=axes[0])
axes[0].set_title('Distribución de Clases (conteo)')
axes[0].set_xlabel('Clase')
axes[0].set_ylabel('Número de muestras')

axes[1].pie(
    counts,
    labels=counts.index,
    autopct='%1.2f%%',
    startangle=90
)
axes[1].set_title('Proporción de Clases (%)')

plt.suptitle(
    f'Análisis de desbalance — {dataset_name} / {target_column}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.show()