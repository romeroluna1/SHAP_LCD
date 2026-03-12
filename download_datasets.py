import pandas as pd
import json
from pandas.api.types import is_object_dtype
from sklearn.datasets import load_iris, load_wine, fetch_openml

REGISTRY_PATH = 'datasets_registry.json'

sklearnLoaders = {
    'vehicle': lambda: fetch_openml(data_id=54, as_frame=False, parser='auto'),
    'iris': lambda: load_iris(),
    'wine': lambda: load_wine(),
}


def save_class_names(alias, class_names):
    path = f'CLASS_NAMES_{alias.upper()}.json'
    with open(path, 'w') as f:
        json.dump({str(k): str(v) for k, v in class_names.items()}, f)
    print(f'Class names guardados en {path}')


with open(REGISTRY_PATH, 'r') as f:
    registry = json.load(f)

for alias, config in registry.items():
    target_col = config.get('target_column', 'target')
    dataset_type = config.get('type', 'custom')

    if dataset_type == 'sklearn':
        dataset = sklearnLoaders[alias]()
        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        df[target_col] = dataset.target

        if is_object_dtype(df[target_col]) or isinstance(df[target_col].dtype, pd.CategoricalDtype):
            categorical = df[target_col].astype('category')
            class_names = {code: name_ for code, name_ in enumerate(categorical.cat.categories)}
            df[target_col] = categorical.cat.codes
            
        elif hasattr(dataset, 'target_names'):
            class_names = {i: str(n) for i, n in enumerate(dataset.target_names)}

        save_class_names(alias, class_names)
        csv_path = config['csv']
        df.to_csv(csv_path, index=False)
        print(f'CSV guardado como {csv_path}')

    elif dataset_type == 'custom':
        sep = config.get('sep', ',')
        csv_path = config['csv']
        df = pd.read_csv(csv_path, sep=sep)

        if is_object_dtype(df[target_col]) or isinstance(df[target_col].dtype, pd.CategoricalDtype):
            categorical = df[target_col].astype('category')
            class_names = {code: name_ for code, name_ in enumerate(categorical.cat.categories)}
            df[target_col] = categorical.cat.codes
            df.to_csv(csv_path, sep=sep, index=False)
        else:
            unique_vals = sorted(df[target_col].unique())
            class_names = {int(v): str(v) for v in unique_vals}

        save_class_names(alias, class_names)
        print(f'Dataset custom "{alias}" procesado (CSV: {csv_path})')
