from SHAP_LCD.SHAPE_Explainer import SHAPExplainer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


CSV_PATH = "iris.csv"
TARGET_COLUMN = "target"
INSTANCE_INDEX = 10
SHAP_FILE = "shap_values_iris_logistic.pickle"
NEW_INSTANCE = [5.0, 2.0, 5.1, 1.8]

df = pd.read_csv(CSV_PATH)
CLASS_NAMES = {i: name for i, name in enumerate(["setosa", "versicolor", "virginica"])}

if __name__ == "__main__":
    explainer = SHAPExplainer(CSV_PATH, LogisticRegression, TARGET_COLUMN)
    explainer.train_and_evaluate()
    explainer.calculate_shap_values()
    explainer.save_shap_values(SHAP_FILE)

    prob_diffs = explainer.plot_probability_differences(INSTANCE_INDEX, "test", CLASS_NAMES, f"Instancia {INSTANCE_INDEX}")
    for i, (diff, c1, c2) in enumerate(prob_diffs):
        print(f"{i+1}. {c1} vs {c2}: diferencia = {diff:.4f}")
    user_input = input("Opción(es): ")
    selected_indices = [int(i.strip()) - 1 for i in user_input.split(",") if i.strip().isdigit()]
    selected_pairs = [(prob_diffs[i][1], prob_diffs[i][2]) for i in selected_indices if 0 <= i < len(prob_diffs)]
    explainer.explain_selected_pairs(INSTANCE_INDEX, selected_pairs, "test", CLASS_NAMES, f"Instancia {INSTANCE_INDEX}")

    new_diffs = explainer.plot_probability_differences(NEW_INSTANCE, "new", CLASS_NAMES, "Instancia nueva")
    for i, (diff, c1, c2) in enumerate(new_diffs):
        print(f"{i+1}. {c1} vs {c2}: diferencia = {diff:.4f}")
    # Solicitar al usuario que seleccione uno o varios pares por número
    print("\nIngrese los números de los pares que desea analizar (ej. 1 o 1,2):")
    new_input = input("Opción(es): ")
    selected_indices_ext = [int(i.strip()) - 1 for i in new_input.split(",") if i.strip().isdigit()]
    selected_new_pairs = [(new_diffs[i][1], new_diffs[i][2]) for i in selected_indices_ext if 0 <= i < len(new_diffs)]
    explainer.explain_selected_pairs(NEW_INSTANCE, selected_new_pairs, "new", CLASS_NAMES, "Instancia nueva")
