from SHAP_LCD.SHAPE_Explainer import SHAPExplainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# --------------------------------------------
# PARÁMETROS GENERALES DEL PROGRAMA
# --------------------------------------------

CSV_PATH = "iris.csv"  # Ruta al archivo CSV del dataset
TARGET_COLUMN = "target"  # Nombre de la columna objetivo (variable de salida)
INSTANCE_INDEX = 10  # Índice de la instancia a analizar del conjunto de prueba
SHAP_FILE = "shap_values_iris.pickle"  # Nombre del archivo para guardar los valores SHAP
NEW_INSTANCE = [5.0, 2.0, 5.1, 1.8]  # Nueva instancia (fuera del dataset original)

# --------------------------------------------
# DEFINICIÓN DE LOS NOMBRES DE CLASE PARA VISUALIZACIÓN
# --------------------------------------------

df = pd.read_csv(CSV_PATH)
y = df[TARGET_COLUMN]
CLASS_NAMES = {i: name for i, name in enumerate(["setosa", "versicolor", "virginica"])}  # Asignación explícita

# --------------------------------------------
# BLOQUE PRINCIPAL DEL PROGRAMA
# --------------------------------------------

if __name__ == "__main__":
    # Inicializar el explicador SHAP con el modelo y datos
    explainer = SHAPExplainer(CSV_PATH, RandomForestClassifier, TARGET_COLUMN)

    # Entrenar modelo, calcular y guardar valores SHAP
    explainer.train_and_evaluate()
    explainer.calculate_shap_values()
    explainer.save_shap_values(SHAP_FILE)

    print("\nX_test shape:", explainer.X_test.shape)
    print("SHAP values shape:", np.array(explainer.shap_values).shape)

    # --------------------------------------------
    # ANÁLISIS DE INSTANCIA DEL CONJUNTO DE PRUEBA
    # --------------------------------------------

    print("\n--- Algoritmo 1: Diferencia de probabilidad entre clases (instancia del conjunto de prueba) ---")
    prob_diffs = explainer.plot_probability_differences(
        INSTANCE_INDEX,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instancia {INSTANCE_INDEX}"
    )

    # Mostrar lista de pares ordenados por diferencia de probabilidad
    for i, (diff, c1, c2) in enumerate(prob_diffs):
        print(f"{i+1}. {c1} vs {c2}: diferencia = {diff:.4f}")

    # Solicitar al usuario que seleccione uno o varios pares por número
    print("\nIngrese los números de los pares que desea analizar (ej. 1 o 1,2):")
    user_input = input("Opción(es): ")

    selected_indices = [int(i.strip()) - 1 for i in user_input.split(",") if i.strip().isdigit()]
    selected_pairs = [(prob_diffs[i][1], prob_diffs[i][2]) for i in selected_indices if 0 <= i < len(prob_diffs)]

    # Mostrar diferencias SHAP para los pares seleccionados
    print("\n--- Algoritmo 2: Diferencias SHAP para pares seleccionados ---")
    explainer.explain_selected_pairs(
        INSTANCE_INDEX,
        selected_pairs=selected_pairs,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instancia {INSTANCE_INDEX}"
    )

    # --------------------------------------------
    # ANÁLISIS DE NUEVA INSTANCIA EXTERNA
    # --------------------------------------------

    print("\n--- Algoritmo 1: Diferencia de probabilidad para instancia externa ---")
    new_diffs = explainer.plot_probability_differences(
        NEW_INSTANCE,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva"
    )

    for i, (diff, c1, c2) in enumerate(new_diffs):
        print(f"{i+1}. {c1} vs {c2}: diferencia = {diff:.4f}")

    print("\nIngrese los números de los pares que desea analizar para la instancia externa (ej. 1 o 1,2):")
    new_input = input("Opción(es): ")

    selected_indices_ext = [int(i.strip()) - 1 for i in new_input.split(",") if i.strip().isdigit()]
    selected_new_pairs = [(new_diffs[i][1], new_diffs[i][2]) for i in selected_indices_ext if 0 <= i < len(new_diffs)]

    print("\n--- Algoritmo 2: Diferencias SHAP para la instancia externa ---")
    explainer.explain_selected_pairs(
        NEW_INSTANCE,
        selected_pairs=selected_new_pairs,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva"
    )
