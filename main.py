from functools import partial
from SHAPE_Explainer import SHAPExplainer
from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

TARGET_COLUMN = 'target'
INSTANCE_INDEX = 10

listDatasets = {
    'vehicle': lambda: fetch_openml(data_id=54, as_frame=False, parser='auto'),
    'iris': lambda: load_iris(),
    'wine': lambda: load_wine(),
}

listAlgorithms = {
    'svc': partial(SVC, probability=True),
    'randomforest': RandomForestClassifier,
    'mlp': partial(MLPClassifier, max_iter=1000, random_state=42),
    'logistic': LogisticRegression,
}

listNewInstancesForDataset = {
    'vehicle': [90, 40, 100, 180, 60, 9, 200, 50, 20, 150, 250, 500, 180, 70, 8, 10, 200, 208],
    'iris': [5.0, 2.0, 5.1, 1.8],
    'wine': [13.0, 1.80, 2.43, 16.0, 102.0, 2.86, 3.03, 0.30, 2.30, 6.50, 1.04, 3.80, 1280.0],
}

dataset = list(listDatasets)[1]
algorithm = list(listAlgorithms)[0]
NEW_INSTANCE = listNewInstancesForDataset[dataset]

CSV_PATH = f'{dataset}.csv'
data = listDatasets[dataset]()

CLASS_NAMES = {i: name for i, name in enumerate(data.target_names)}
print('classnames', CLASS_NAMES)
print('dataset', dataset)
print('algorithm', algorithm)
print('new_instance', NEW_INSTANCE)

model_svc = listAlgorithms[algorithm]
SHAP_FILE = f'shap_values_{dataset}_{algorithm}.pickle'


if __name__ == "__main__":
    explainer = SHAPExplainer(CSV_PATH, model_svc, TARGET_COLUMN)
    explainer.train_and_evaluate()
    explainer.calculate_shap_values()
    explainer.save_shap_values(SHAP_FILE)
    prob_diffs = explainer.plot_probability_differences(INSTANCE_INDEX, "test", CLASS_NAMES,
                                                        f"Instancia {INSTANCE_INDEX}")

    selected_pairs = [(prob_diffs[i][1], prob_diffs[i][2]) for i, value in enumerate(prob_diffs) if 0 <= i < len(prob_diffs)]
    explainer.explain_selected_pairs(INSTANCE_INDEX, selected_pairs, "test", CLASS_NAMES, f"Instancia {INSTANCE_INDEX}")

    new_diffs = explainer.plot_probability_differences(NEW_INSTANCE, "new", CLASS_NAMES, "Instancia nueva")

    selected_new_pairs = [(new_diffs[i][1], new_diffs[i][2]) for i, value in enumerate(new_diffs) if 0 <= i < len(new_diffs)]
    explainer.explain_selected_pairs(NEW_INSTANCE, selected_new_pairs, "new", CLASS_NAMES, "Instancia nueva")

