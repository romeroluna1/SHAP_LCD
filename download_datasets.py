import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_openml

# vehicle have two datasets id=54 y id=41147
listDatasets = {
    'vehicle': lambda: fetch_openml(data_id=54, as_frame=False, parser='auto'),
    'iris': lambda: load_iris(),
    'wine': lambda: load_wine()
}

for name, action in listDatasets.items():
    dataset = action()
    nameDataset = f'{name}.csv'
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df.to_csv(nameDataset, index=False)
    print(f'CSV guardado como {nameDataset}')
