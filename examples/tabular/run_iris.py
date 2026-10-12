from pathlib import Path
import os
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd import SHAPExplainer


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO PUBLICADO
# ============================================================

SEED = 42
TARGET_COLUMN = "target"
INSTANCE_INDEX = 10
NEW_INSTANCE = [5.0, 2.0, 5.1, 1.8]

CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

METRIC_PAIR = (1, 2)

# True ejecuta una prueba rápida de integración.
# False reproduce la configuración completa del artículo.
QUICK_TEST = False

M_PERTURB = 2 if QUICK_TEST else 40
N_INSTANCES = 2 if QUICK_TEST else None

SIGMA = 0.02
TOP_K = 10
TAU = 0.95
NORMALIZE = "l2"
SHAP_MODE = "auto"


# ============================================================
# RUTAS DE DATOS, ARTEFACTOS Y RESULTADOS
# ============================================================

CSV_PATH = PROJECT_ROOT / "data" / "iris.csv"
SHAP_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "iris"
    / "shap_values_iris.pickle"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "iris"
RESULTS_DIR = RESULTS_ROOT / ("quick_test" if QUICK_TEST else "final")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_environment():
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("SHAP:", shap.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("seaborn:", sns.__version__)
    print("OS:", platform.platform())
    print("Processor:", platform.processor())


def get_shap_instance(shap_values, instance_index):
    """Extrae los valores SHAP de una instancia para cada clase."""
    if isinstance(shap_values, list):
        return [
            np.asarray(shap_values[c][instance_index])
            for c in range(len(shap_values))
        ]

    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 3:
        raise ValueError(
            f"Forma SHAP inesperada: {shap_array.shape}"
        )

    return [
        shap_array[instance_index, :, c]
        for c in range(shap_array.shape[2])
    ]


def verify_probability_scale(explainer, instance_index):
    """Verifica si el TreeExplainer de RF reconstruye predict_proba."""
    x = np.asarray(explainer.X_test[instance_index]).reshape(1, -1)
    probabilities = explainer.modelo_entrenado_70.predict_proba(x)[0]
    shap_instance = get_shap_instance(
        explainer.shap_values,
        instance_index,
    )
    expected_value = np.asarray(
        explainer.explainer.expected_value
    ).reshape(-1)

    print("\n" + "=" * 72)
    print(" VERIFICACIÓN DE LA ESCALA SHAP - IRIS")
    print("=" * 72)
    print(
        f"{'Clase':16s} {'predict_proba':>16s} "
        f"{'base + sum(SHAP)':>20s} {'error':>13s}"
    )
    print("-" * 72)

    errors = []
    for c, probability in enumerate(probabilities):
        reconstructed = expected_value[c] + np.sum(shap_instance[c])
        error = abs(float(probability) - float(reconstructed))
        errors.append(error)
        print(
            f"{CLASS_NAMES[c]:16s} {probability:16.10f} "
            f"{reconstructed:20.10f} {error:13.3e}"
        )

    max_error = max(errors)
    print("-" * 72)
    print(f"Error máximo de reconstrucción: {max_error:.3e}")

    if max_error >= 1e-5:
        raise RuntimeError(
            "El pickle SHAP no reconstruye predict_proba. "
            "No debe utilizarse para este experimento."
        )

    print("VERIFICACIÓN SUPERADA: SHAP está en probabilidad.")
    print("=" * 72)


def select_pairs(differences, prompt):
    user_input = input(prompt)
    selected_indices = [
        int(value.strip()) - 1
        for value in user_input.split(",")
        if value.strip().isdigit()
    ]
    return [
        (differences[i][1], differences[i][2])
        for i in selected_indices
        if 0 <= i < len(differences)
    ]


def print_statistical_details(df_inst):
    n_instances = len(df_inst)
    mean_s = float(df_inst["Stability_S"].mean())
    mean_c = float(df_inst["Consistency_C"].mean())
    std_s = float(df_inst["Stability_S"].std(ddof=1))
    std_c = float(df_inst["Consistency_C"].std(ddof=1))
    se_s = std_s / np.sqrt(n_instances)
    se_c = std_c / np.sqrt(n_instances)
    z_value = 1.96

    print("\n" + "=" * 70)
    print(" ESTADÍSTICOS PARA JUSTIFICAR N")
    print("=" * 70)
    print(f"N instancias evaluadas: {n_instances}")
    print(f"M perturbaciones por instancia: {M_PERTURB}")
    print(f"Stability_S mean: {mean_s:.6f}")
    print(f"Consistency_C mean: {mean_c:.6f}")
    print(f"s_S: {std_s:.6f}")
    print(f"s_C: {std_c:.6f}")
    print(f"SE(S_mean): {se_s:.6f}")
    print(f"SE(C_mean): {se_c:.6f}")
    print(
        f"CI95(S_mean): "
        f"[{mean_s - z_value * se_s:.6f}, "
        f"{mean_s + z_value * se_s:.6f}]"
    )
    print(
        f"CI95(C_mean): "
        f"[{mean_c - z_value * se_c:.6f}, "
        f"{mean_c + z_value * se_c:.6f}]"
    )
    print("=" * 70)


# ============================================================
# BLOQUE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(RESULTS_DIR)

    print_environment()

    print("\n" + "=" * 72)
    print(" SHAP-LCD - IRIS - RANDOM FOREST")
    print("=" * 72)
    print(f"Dataset       : {CSV_PATH}")
    print(f"SHAP pickle   : {SHAP_FILE}")
    print(f"Resultados    : {RESULTS_DIR}")
    print(f"Quick test    : {QUICK_TEST}")
    print(f"M             : {M_PERTURB}")
    print(f"N instances   : {N_INSTANCES}")
    print("=" * 72)

    explainer = SHAPExplainer(
        csv_path=CSV_PATH,
        model_class=RandomForestClassifier,
        target_column=TARGET_COLUMN,
        random_state=SEED,
        output_scale="native",
    )

    explainer.train_and_evaluate()
    explainer.ensure_shap_values(SHAP_FILE, mode=SHAP_MODE)

    print("\nX_test shape:", explainer.X_test.shape)
    print("SHAP values shape:", np.asarray(explainer.shap_values).shape)

    verify_probability_scale(explainer, INSTANCE_INDEX)

    # --------------------------------------------------------
    # INSTANCIA DEL TEST
    # --------------------------------------------------------
    print("\n--- Algoritmo 1: instancia del test ---")
    prob_diffs = explainer.plot_probability_differences(
        INSTANCE_INDEX,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instance {INSTANCE_INDEX}",
        save_path=(
            RESULTS_DIR
            / f"probability_differences_instance_{INSTANCE_INDEX}.png"
        ),
        dpi=600,
    )

    for i, (difference, class_i, class_j) in enumerate(prob_diffs, start=1):
        print(
            f"{i}. {class_i} vs {class_j}: "
            f"|Delta P| = {difference:.8f}"
        )

    selected_pairs = select_pairs(
        prob_diffs,
        "Opción(es) para la instancia del test (ej. 1 o 1,2): ",
    )

    explainer.explain_selected_pairs(
        INSTANCE_INDEX,
        selected_pairs=selected_pairs,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instance {INSTANCE_INDEX}",
        save_dir=RESULTS_DIR,
        dpi=600,
    )

    # --------------------------------------------------------
    # INSTANCIA EXTERNA
    # --------------------------------------------------------
    print("\n--- Algoritmo 1: instancia externa ---")
    new_diffs = explainer.plot_probability_differences(
        NEW_INSTANCE,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva",
        save_path=(
            RESULTS_DIR
            / "probability_differences_external_instance.png"
        ),
        dpi=600,
    )

    for i, (difference, class_i, class_j) in enumerate(new_diffs, start=1):
        print(
            f"{i}. {class_i} vs {class_j}: "
            f"|Delta P| = {difference:.8f}"
        )

    selected_new_pairs = select_pairs(
        new_diffs,
        "Opción(es) para la instancia externa (ej. 1 o 1,2): ",
    )

    explainer.explain_selected_pairs(
        NEW_INSTANCE,
        selected_pairs=selected_new_pairs,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva",
        save_dir=RESULTS_DIR,
        dpi=600,
    )

    # --------------------------------------------------------
    # FIGURAS GLOBALES
    # --------------------------------------------------------
    explainer.plot_global_violin_by_class(
        class_names=CLASS_NAMES,
        use_normalized=True,
        dataset_name="Iris",
        mode="pooled_features",
        save_path=RESULTS_DIR / "iris_global_violin.png",
        dpi=600,
        figsize=(8, 5),
        violin_alpha=0.75,
        fliersize=8,
    )

    explainer.plot_class_distribution_bar(
        class_names=CLASS_NAMES,
        dataset_name="Iris",
        save_path=RESULTS_DIR / "iris_class_distribution.png",
        dpi=600,
        figsize=(8, 5),
        cmap_min=0.2,
        cmap_max=0.8,
        show_values=True,
    )

    # --------------------------------------------------------
    # MÉTRICAS: UNA SOLA PASADA
    # --------------------------------------------------------
    print("\n--- Evaluación SHAP-LCD: versicolor vs virginica ---")
    df_cmp, summary_cmp = (
        explainer.evaluate_shap_lcd_pair_compare_consistency(
            pair=METRIC_PAIR,
            class_names=CLASS_NAMES,
            filter_mode="true",
            n_instances=N_INSTANCES,
            M=M_PERTURB,
            sigma=SIGMA,
            top_k=TOP_K,
            tau=TAU,
            normalize=NORMALIZE,
            seed=SEED,
        )
    )

    df_inst = df_cmp[
        [
            "idx_test",
            "pair",
            "abs_deltaP_ij",
            "Stability_S",
            "Consistency_C_predexp",
        ]
    ].copy()
    df_inst = df_inst.rename(
        columns={"Consistency_C_predexp": "Consistency_C"}
    )

    summary = (
        df_inst[["Stability_S", "Consistency_C"]]
        .agg(["mean", "std"])
        .T
    )
    summary["mean±std"] = (
        summary["mean"].map(lambda value: f"{value:.6f}")
        + " ± "
        + summary["std"].map(lambda value: f"{value:.6f}")
    )

    print("\nResumen original:")
    print(summary[["mean±std"]].to_string())
    print("\nResumen completo:")
    print(summary_cmp[["mean±std"]].to_string())

    indices = df_inst["idx_test"].to_numpy(dtype=int)
    print("\ny_test counts:")
    print(
        pd.Series(explainer.y_test[indices])
        .value_counts()
        .sort_index()
    )

    df_inst.to_csv(
        RESULTS_DIR / "iris_shap_lcd_pair_metrics_1_2.csv",
        index=False,
    )
    df_cmp.to_csv(
        RESULTS_DIR / "iris_compare_consistency_1_2.csv",
        index=False,
    )

    print_statistical_details(df_inst)

    print("\n" + "=" * 72)
    print(" EJECUCIÓN DE IRIS FINALIZADA")
    print("=" * 72)
    print(f"Resultados guardados en: {RESULTS_DIR}")
    print("=" * 72)
