from SHAP_LCD.SHAPE_Explainer import SHAPExplainer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

import sys
import sklearn
import shap
import matplotlib
import seaborn as sns
import platform

# --------------------------------------------
# INFO DE ENTORNO (igual que Iris/Wine)
# --------------------------------------------
print("Python:", sys.version)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("OS:", platform.platform())
print("Processor:", platform.processor())

# --------------------------------------------
# PARÁMETROS GENERALES
# --------------------------------------------
CSV_PATH = r"C:\Users\romer\OneDrive\Desktop\Roxana Romero\ExplicabilidadRF\Datos_vehicule\statlog_vehicle_clean.csv"
TARGET_COLUMN = "class_num"
INSTANCE_INDEX = 10
SHAP_FILE = "shap_values_vehicle_dt.pickle"

# --------------------------------------------
# NOMBRES DE CLASE (idx -> nombre)
# --------------------------------------------
CLASS_NAMES = {
    0: "bus",
    1: "saab",
    2: "opel",
    3: "van",
}

# --------------------------------------------
# NUEVA INSTANCIA EXTERNA (AUTOAJUSTADA A #FEATURES)
# --------------------------------------------
df = pd.read_csv(CSV_PATH)
feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
NEW_INSTANCE = [0.0] * len(feature_cols)

# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    # 1) Inicializar explicador
    #    IMPORTANTE: aquí NO usamos target_as_str=True para que las clases sean int (0,1,2,3)
    #    y se puedan mapear bien a CLASS_NAMES en las métricas.
    explainer = SHAPExplainer(
        CSV_PATH,
        lambda **kwargs: DecisionTreeClassifier(random_state=42),
        TARGET_COLUMN,
        target_as_str=False
    )

    # 2) Entrenar modelo (CV + modelo final 70%)
    explainer.train_and_evaluate()

    # 3) SHAP: reutilizar si existe; si no existe calcula+guarda
    explainer.ensure_shap_values(SHAP_FILE, mode="auto")

    print("\nX_test shape:", explainer.X_test.shape)
    print("SHAP values shape:", np.array(explainer.shap_values).shape)

    # --------------------------------------------
    # ANÁLISIS DE INSTANCIA DEL TEST
    # --------------------------------------------
    print("\n--- Algoritmo 1: Diferencia de probabilidad entre clases (instancia del test) ---")
    prob_diffs = explainer.plot_probability_differences(
        INSTANCE_INDEX,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instancia {INSTANCE_INDEX}"
    )

    for i, (diff, c1, c2) in enumerate(prob_diffs):
        print(f"{i+1}. {c1} vs {c2}: diferencia = {diff:.4f}")

    print("\nIngrese los números de los pares que desea analizar (ej. 1 o 1,2):")
    user_input = input("Opción(es): ")
    selected_indices = [int(i.strip()) - 1 for i in user_input.split(",") if i.strip().isdigit()]
    selected_pairs = [(prob_diffs[i][1], prob_diffs[i][2]) for i in selected_indices if 0 <= i < len(prob_diffs)]

    print("\n--- Algoritmo 2: Diferencias SHAP para pares seleccionados (test) ---")
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
    print("\n--- Algoritmo 1: Diferencia de probabilidad (instancia externa) ---")
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

    print("\n--- Algoritmo 2: Diferencias SHAP para pares seleccionados (instancia externa) ---")
    explainer.explain_selected_pairs(
        NEW_INSTANCE,
        selected_pairs=selected_new_pairs,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva"
    )

    # --------------------------------------------
    # FIGURAS GLOBALES
    # --------------------------------------------
    explainer.plot_global_violin_by_class(
        class_names=CLASS_NAMES,
        use_normalized=True,
        dataset_name="Vehicle",
        mode="pooled_features",
        save_path="vehicle_global_violin.png",
        dpi=600,
        figsize=(10, 5),
        violin_alpha=0.75,
        fliersize=8
    )

    explainer.plot_class_distribution_bar(
        class_names=CLASS_NAMES,
        dataset_name="Vehicle",
        save_path="vehicle_class_distribution.png",
        dpi=600,
        figsize=(8, 5),
        cmap_min=0.2,
        cmap_max=0.8,
        show_values=True
    )

    # --------------------------------------------
    # MÉTRICAS SHAP-LCD (par fijo) — con nombres de clases
    # --------------------------------------------
    print("\n--- Evaluación SHAP_LCD (par fijo) ---")
    df_inst, summary = explainer.evaluate_shap_lcd_pair(
        pair=(0, 1),                 # ✅ enteros
        class_names=CLASS_NAMES,     # ✅ para que salga "bus vs saab"
        filter_mode="true",
        n_instances=None,
        M=40,
        sigma=0.02,
        top_k=10,
        tau=0.95,
        normalize="l2",
        seed=42
    )

    print("\nResumen (media ± std):")
    print(summary[["mean±std"]].to_string())

    print("\nPrimeras filas:")
    print(df_inst.head(10).to_string(index=False))

    df_inst.to_csv("vehicle_shap_lcd_pair_metrics_0_1.csv", index=False)

    # --------------------------------------------
    # COMPARACIÓN DE CONSISTENCIA (dos versiones)
    # --------------------------------------------
    print("\n--- Comparación de consistencia (pred->exp vs exp->pred) ---")
    df_cmp, summary_cmp = explainer.evaluate_shap_lcd_pair_compare_consistency(
        pair=(0, 1),                 # ✅ enteros
        class_names=CLASS_NAMES,     # ✅ nombres
        filter_mode="true",
        n_instances=None,
        M=40,
        sigma=0.02,
        top_k=10,
        tau=0.95,
        normalize="l2",
        seed=42
    )

    print("\nResumen comparación (media ± std):")
    print(summary_cmp[["mean±std"]].to_string())

    print("\nPrimeras filas df_cmp:")
    print(df_cmp.head().to_string(index=False))

    df_cmp.to_csv("vehicle_compare_consistency_0_1.csv", index=False)


print("N instancias evaluadas:", len(df_inst))
print("M (perturbaciones por instancia):", 40)  # o el M que estés usando
print("\nPromedios globales (sobre instancias):")
print("Stability_S mean:", df_inst["Stability_S"].mean())
print("Consistency_C mean:", df_inst["Consistency_C"].mean())
idxs = df_inst["idx_test"].values
print("y_test counts en esas instancias:")
print(pd.Series(explainer.y_test[idxs]).value_counts().sort_index())



# N (instancias evaluadas)
N = len(df_inst)

# Medias (promedio global sobre instancias)
mean_S = float(df_inst["Stability_S"].mean())
mean_C = float(df_inst["Consistency_C"].mean())

# Desv. estándar entre instancias (s_S y s_C)
s_S = float(df_inst["Stability_S"].std(ddof=1))
s_C = float(df_inst["Consistency_C"].std(ddof=1))

# Error estándar del promedio (SE)
SE_S = s_S / np.sqrt(N)
SE_C = s_C / np.sqrt(N)

print("\n==============================")
print(" Estadísticos para justificar N")
print("==============================")
print(f"N instancias evaluadas: {N}")
print(f"M (perturbaciones por instancia): {40}")  # cambia si usas otro M
print("\n--- Promedios globales (sobre instancias) ---")
print(f"Stability_S mean: {mean_S:.6f}")
print(f"Consistency_C mean: {mean_C:.6f}")
print("\n--- Dispersión entre instancias (std) ---")
print(f"s_S (std Stability_S): {s_S:.6f}")
print(f"s_C (std Consistency_C): {s_C:.6f}")
print("\n--- Error estándar del promedio ---")
print(f"SE(S_mean) = s_S/sqrt(N): {SE_S:.6f}")
print(f"SE(C_mean) = s_C/sqrt(N): {SE_C:.6f}")
print("==============================\n")
z = 1.96  # aproximación normal
ciS_low, ciS_high = mean_S - z * SE_S, mean_S + z * SE_S
ciC_low, ciC_high = mean_C - z * SE_C, mean_C + z * SE_C

print("\n--- Intervalo de confianza aproximado (95%) ---")
print(f"CI95(S_mean): [{ciS_low:.6f}, {ciS_high:.6f}]")
print(f"CI95(C_mean): [{ciC_low:.6f}, {ciC_high:.6f}]")
