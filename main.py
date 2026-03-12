from functools import partial
from SHAPE_Explainer import SHAPExplainer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
import json
import os

TARGET_COLUMN = 'target'
INSTANCE_INDEX = 10

REGISTRY_PATH = 'datasets_registry.json'
with open(REGISTRY_PATH, 'r') as f:
    datasets_registry = json.load(f)

listDatasets = list(datasets_registry.keys())

listAlgorithms = {
    'svc': partial(SVC, probability=True),
    'randomforest': RandomForestClassifier,
    'mlp': partial(MLPClassifier, max_iter=1000, random_state=42),
    'logistic': LogisticRegression,
}

dataset_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) < len(listDatasets) else 1
algorithm_index = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() and int(sys.argv[2]) < len(listAlgorithms) else 0

dataset = listDatasets[dataset_index]
algorithm = list(listAlgorithms)[algorithm_index]

dataset_config = datasets_registry[dataset]
CSV_PATH = dataset_config['csv']
TARGET_COLUMN = dataset_config.get('target_column', 'target')
NEW_INSTANCE = dataset_config.get('new_instance')
CLASS_NAMES_PATH = f'CLASS_NAMES_{dataset.upper()}.json'

if not os.path.exists(CSV_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Archivos de dataset no encontrados para '{dataset}'. Ejecuta primero: python download_datasets.py"
    )

if NEW_INSTANCE is None:
    raise ValueError(
        f"'{dataset}' no tiene 'new_instance' definido en {REGISTRY_PATH}."
    )

with open(CLASS_NAMES_PATH, 'r') as f:
    CLASS_NAMES = {int(k): v for k, v in json.load(f).items()}


print('classnames', CLASS_NAMES)
print('dataset', dataset)
print('algorithm', algorithm)
print('new_instance', NEW_INSTANCE)

SHAP_FILE = f'shap_values_{dataset}_{algorithm}.pickle'
PATH_BASE_IMAGES = f'images_{dataset}_{algorithm}'

def main(model):
    sep = dataset_config.get('sep', ',')
    explainer = SHAPExplainer(CSV_PATH, model, TARGET_COLUMN, PATH_BASE_IMAGES, sep=sep)
    explainer.load_or_compute(SHAP_FILE)
    prob_diffs = explainer.plot_probability_differences(INSTANCE_INDEX, "test", CLASS_NAMES,
                                                        f"Instancia {INSTANCE_INDEX}")

    if not prob_diffs:
        return

    selected_pairs = [(prob_diffs[i][1], prob_diffs[i][2]) for i, value in enumerate(prob_diffs) if 0 <= i < len(prob_diffs)]
    explainer.explain_selected_pairs(INSTANCE_INDEX, selected_pairs, "test", CLASS_NAMES, f"Instancia {INSTANCE_INDEX}")

    new_diffs = explainer.plot_probability_differences(NEW_INSTANCE, "new", CLASS_NAMES, "Instancia nueva") if NEW_INSTANCE is not None else None

    if not new_diffs:
        return

    selected_new_pairs = [(new_diffs[i][1], new_diffs[i][2]) for i, value in enumerate(new_diffs) if 0 <= i < len(new_diffs)]
    explainer.explain_selected_pairs(NEW_INSTANCE, selected_new_pairs, "new", CLASS_NAMES, "Instancia nueva")

if __name__ == "__main__":
    model_choose = listAlgorithms[algorithm]
    main(model_choose)
