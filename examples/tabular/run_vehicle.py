from pathlib import Path
import os
import sys
import platform

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1) RUTAS DEL PROYECTO
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd.explainer import SHAPExplainer


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "statlog_vehicle_clean.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "vehicle"
)

QUICK_TEST = False

if QUICK_TEST:
    OUTPUT_DIR = (
        PROJECT_ROOT
        / "results"
        / "vehicle"
        / "quick_test"
    )
else:
    OUTPUT_DIR = (
        PROJECT_ROOT
        / "results"
        / "vehicle"
        / "final"
    )

SHAP_FILE = (
    ARTIFACT_DIR
    / "shap_values_vehicle_dt.pickle"
)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Los archivos generados sin ruta absoluta también quedarán aquí.
os.chdir(OUTPUT_DIR)


# ============================================================
# 2) INFORMACIÓN DEL ENTORNO
# ============================================================
print("Python:", sys.version)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
print("OS:", platform.platform())
print("Processor:", platform.processor())


# ============================================================
# 3) CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================
TARGET_COLUMN = "class_num"
INSTANCE_INDEX = 10

CLASS_NAMES = {
    0: "bus",
    1: "saab",
    2: "opel",
    3: "van",
}

# Par utilizado en el análisis local presentado en el artículo.
LOCAL_PAIR = (0, 3)  # bus vs van

# Pares utilizados en la evaluación de estabilidad y consistencia.
EVALUATION_PAIRS = [
    (0, 1),  # bus vs saab
    (1, 2),  # saab vs opel
]

M_FINAL = 40
N_FINAL = None

M_QUICK = 2
N_QUICK = 2

M = M_QUICK if QUICK_TEST else M_FINAL
N_INSTANCES = N_QUICK if QUICK_TEST else N_FINAL

SIGMA = 0.02
TOP_K = 10
TAU = 0.95
NORMALIZE = "l2"
SEED = 42
FILTER_MODE = "true"


# ============================================================
# 4) FUNCIONES AUXILIARES
# ============================================================
def get_class_shap(shap_values, instance_index, class_index):
    """
    Extrae los valores SHAP de una instancia y una clase,
    admitiendo las principales estructuras retornadas por SHAP.
    """
    if isinstance(shap_values, list):
        return np.asarray(
            shap_values[class_index][instance_index],
            dtype=float
        )

    values = np.asarray(shap_values)

    if values.ndim != 3:
        raise ValueError(
            "Se esperaba una estructura SHAP tridimensional."
        )

    # Formato: (N, características, clases)
    if values.shape[2] == len(CLASS_NAMES):
        return np.asarray(
            values[instance_index, :, class_index],
            dtype=float
        )

    # Formato: (clases, N, características)
    if values.shape[0] == len(CLASS_NAMES):
        return np.asarray(
            values[class_index, instance_index, :],
            dtype=float
        )

    raise ValueError(
        f"No se reconoce la forma SHAP: {values.shape}"
    )


def get_expected_value(expected_value, class_index):
    """
    Obtiene el valor esperado correspondiente a una clase.
    """
    values = np.asarray(expected_value)

    if values.ndim == 0:
        return float(values)

    values = values.reshape(-1)

    if class_index >= len(values):
        raise IndexError(
            "El índice de clase no está disponible en expected_value."
        )

    return float(values[class_index])


def verify_probability_scale(explainer, instance_index=0):
    """
    Comprueba que expected_value + suma(SHAP) reconstruya
    predict_proba para cada clase.
    """
    x = explainer.X_test[instance_index].reshape(1, -1)

    probabilities = np.asarray(
        explainer.modelo_entrenado_70.predict_proba(x)[0],
        dtype=float
    )

    print("\n" + "=" * 72)
    print(" VERIFICACIÓN DE LA ESCALA SHAP - VEHICLE")
    print("=" * 72)
    print(
        f"{'Clase':<15}"
        f"{'predict_proba':>20}"
        f"{'base + sum(SHAP)':>22}"
        f"{'error':>14}"
    )
    print("-" * 72)

    errors = []

    for class_index, probability in enumerate(probabilities):
        class_shap = get_class_shap(
            explainer.shap_values,
            instance_index,
            class_index
        )

        base_value = get_expected_value(
            explainer.explainer.expected_value,
            class_index
        )

        reconstructed = base_value + float(np.sum(class_shap))
        error = abs(float(probability) - reconstructed)
        errors.append(error)

        print(
            f"{CLASS_NAMES[class_index]:<15}"
            f"{probability:>20.10f}"
            f"{reconstructed:>22.10f}"
            f"{error:>14.3e}"
        )

    max_error = float(np.max(errors))

    print("-" * 72)
    print(f"Error máximo de reconstrucción: {max_error:.3e}")

    if max_error <= 1e-6:
        print(
            "VERIFICACIÓN SUPERADA: SHAP está en probabilidad."
        )
    else:
        raise RuntimeError(
            "La reconstrucción SHAP no coincide con predict_proba. "
            "No continúe con la evaluación."
        )

    print("=" * 72)


def select_first_pair(probability_differences):
    """
    Selecciona automáticamente el par con mayor |Delta P|.
    """
    if len(probability_differences) == 0:
        raise ValueError(
            "No se encontraron pares de clases."
        )

    first = probability_differences[0]
    return [(first[1], first[2])]


def print_statistical_summary(df_inst, m_value, pair_name):
    """
    Presenta media, desviación estándar, error estándar
    e intervalo normal aproximado del 95%.
    """
    n = len(df_inst)

    mean_s = float(df_inst["Stability_S"].mean())
    mean_c = float(df_inst["Consistency_C"].mean())

    std_s = float(df_inst["Stability_S"].std(ddof=1))
    std_c = float(df_inst["Consistency_C"].std(ddof=1))

    se_s = std_s / np.sqrt(n)
    se_c = std_c / np.sqrt(n)

    z_value = 1.96

    ci_s_low = max(0.0, mean_s - z_value * se_s)
    ci_s_high = mean_s + z_value * se_s

    ci_c_low = max(0.0, mean_c - z_value * se_c)
    ci_c_high = min(1.0, mean_c + z_value * se_c)

    print("\n" + "=" * 72)
    print(f" ESTADÍSTICOS PARA JUSTIFICAR N - {pair_name}")
    print("=" * 72)
    print(f"N instancias evaluadas: {n}")
    print(f"M perturbaciones por instancia: {m_value}")

    print("\nPromedios globales:")
    print(f"Stability_S mean: {mean_s:.6f}")
    print(f"Consistency_C mean: {mean_c:.6f}")

    print("\nDispersión entre instancias:")
    print(f"s_S: {std_s:.6f}")
    print(f"s_C: {std_c:.6f}")

    print("\nError estándar:")
    print(f"SE(S_mean): {se_s:.6f}")
    print(f"SE(C_mean): {se_c:.6f}")

    print("\nIntervalos aproximados del 95%:")
    print(
        f"CI95(S_mean): "
        f"[{ci_s_low:.6f}, {ci_s_high:.6f}]"
    )
    print(
        f"CI95(C_mean): "
        f"[{ci_c_low:.6f}, {ci_c_high:.6f}]"
    )
    print("=" * 72)


# ============================================================
# 5) PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":

    print("\n" + "=" * 72)
    print(" SHAP-LCD - VEHICLE - DECISION TREE")
    print("=" * 72)
    print("Dataset       :", DATA_PATH)
    print("SHAP pickle   :", SHAP_FILE)
    print("Resultados    :", OUTPUT_DIR)
    print("Quick test    :", QUICK_TEST)
    print("Par local     : bus vs van")
    print("Pares métricas: bus vs saab; saab vs opel")
    print("M             :", M)
    print("N instances   :", N_INSTANCES)
    print("sigma         :", SIGMA)
    print("Top-K         :", TOP_K)
    print("tau           :", TAU)
    print("Filtro        :", FILTER_MODE)
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{DATA_PATH}"
        )

    # --------------------------------------------------------
    # Leer datos para construir la instancia externa
    # --------------------------------------------------------
    data_frame = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data_frame.columns:
        raise ValueError(
            f"No existe la columna objetivo '{TARGET_COLUMN}'."
        )

    feature_columns = [
        column
        for column in data_frame.columns
        if column != TARGET_COLUMN
    ]

    NEW_INSTANCE = [0.0] * len(feature_columns)

    # --------------------------------------------------------
    # Inicializar SHAPExplainer
    # --------------------------------------------------------
    explainer = SHAPExplainer(
        csv_path=str(DATA_PATH),
        model_class=lambda **kwargs: DecisionTreeClassifier(
            random_state=42
        ),
        target_column=TARGET_COLUMN,
        target_as_str=False,
        output_scale="native"
    )

    # --------------------------------------------------------
    # Entrenamiento y validación cruzada
    # --------------------------------------------------------
    explainer.train_and_evaluate()

    # --------------------------------------------------------
    # Cargar o calcular SHAP
    # --------------------------------------------------------
    explainer.ensure_shap_values(
        str(SHAP_FILE),
        mode="auto"
    )

    print("\nX_test shape:", explainer.X_test.shape)
    print(
        "SHAP values shape:",
        np.asarray(explainer.shap_values).shape
    )

    # --------------------------------------------------------
    # Verificar que SHAP reconstruya predict_proba
    # --------------------------------------------------------
    verify_probability_scale(
        explainer,
        instance_index=INSTANCE_INDEX
    )

    # --------------------------------------------------------
    # Algoritmo 1: instancia del test
    # --------------------------------------------------------
    print(
        "\n--- Algoritmo 1: instancia del test ---"
    )

    probability_differences = (
        explainer.plot_probability_differences(
            INSTANCE_INDEX,
            instance_type="test",
            class_names=CLASS_NAMES,
            instance_label=f"Instancia {INSTANCE_INDEX}",
            save_path=str(
                OUTPUT_DIR
                / f"probability_differences_instance_"
                  f"{INSTANCE_INDEX}.png"
            )
        )
    )

    for index, (difference, class_1, class_2) in enumerate(
        probability_differences,
        start=1
    ):
        print(
            f"{index}. {class_1} vs {class_2}: "
            f"|Delta P| = {difference:.8f}"
        )

    selected_pairs = [(
        CLASS_NAMES[LOCAL_PAIR[0]],
        CLASS_NAMES[LOCAL_PAIR[1]]
    )]

    print(
        "\nPar seleccionado automáticamente:",
        selected_pairs[0][0],
        "vs",
        selected_pairs[0][1]
    )

    # --------------------------------------------------------
    # Algoritmo 2: instancia del test
    # --------------------------------------------------------
    explainer.explain_selected_pairs(
        INSTANCE_INDEX,
        selected_pairs=selected_pairs,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Instancia {INSTANCE_INDEX}",
        save_dir=str(OUTPUT_DIR)
    )

    # --------------------------------------------------------
    # Algoritmo 1: instancia externa
    # --------------------------------------------------------
    print(
        "\n--- Algoritmo 1: instancia externa ---"
    )

    external_differences = (
        explainer.plot_probability_differences(
            NEW_INSTANCE,
            instance_type="new",
            class_names=CLASS_NAMES,
            instance_label="Instancia nueva",
            save_path=str(
                OUTPUT_DIR
                / "probability_differences_external_instance.png"
            )
        )
    )

    for index, (difference, class_1, class_2) in enumerate(
        external_differences,
        start=1
    ):
        print(
            f"{index}. {class_1} vs {class_2}: "
            f"|Delta P| = {difference:.8f}"
        )

    selected_external_pairs = [(
        CLASS_NAMES[LOCAL_PAIR[0]],
        CLASS_NAMES[LOCAL_PAIR[1]]
    )]

    print(
        "\nPar seleccionado automáticamente:",
        selected_external_pairs[0][0],
        "vs",
        selected_external_pairs[0][1]
    )

    # --------------------------------------------------------
    # Algoritmo 2: instancia externa
    # --------------------------------------------------------
    explainer.explain_selected_pairs(
        NEW_INSTANCE,
        selected_pairs=selected_external_pairs,
        instance_type="new",
        class_names=CLASS_NAMES,
        instance_label="Instancia nueva",
        save_dir=str(OUTPUT_DIR)
    )

    # --------------------------------------------------------
    # Figuras globales complementarias
    # --------------------------------------------------------
    explainer.plot_global_violin_by_class(
        class_names=CLASS_NAMES,
        use_normalized=True,
        dataset_name="Vehicle",
        mode="pooled_features",
        save_path=str(
            OUTPUT_DIR
            / "vehicle_global_violin.png"
        ),
        dpi=600,
        figsize=(10, 5),
        violin_alpha=0.75,
        fliersize=8
    )

    explainer.plot_class_distribution_bar(
        class_names=CLASS_NAMES,
        dataset_name="Vehicle",
        save_path=str(
            OUTPUT_DIR
            / "vehicle_class_distribution.png"
        ),
        dpi=600,
        figsize=(8, 5),
        cmap_min=0.2,
        cmap_max=0.8,
        show_values=True
    )

    # --------------------------------------------------------
    # Estabilidad y consistencia para los pares publicados
    # --------------------------------------------------------
    all_original_results = []
    all_comparison_results = []
    final_summary_rows = []

    for class_j, class_r in EVALUATION_PAIRS:
        pair = (class_j, class_r)
        pair_name = (
            f"{CLASS_NAMES[class_j]} vs {CLASS_NAMES[class_r]}"
        )
        pair_tag = f"{class_j}_{class_r}"

        print("\n" + "#" * 72)
        print(f" MÉTRICAS SHAP-LCD: {pair_name}")
        print("#" * 72)

        df_inst, summary = explainer.evaluate_shap_lcd_pair(
            pair=pair,
            class_names=CLASS_NAMES,
            filter_mode=FILTER_MODE,
            n_instances=N_INSTANCES,
            M=M,
            sigma=SIGMA,
            top_k=TOP_K,
            tau=TAU,
            normalize=NORMALIZE,
            seed=SEED
        )

        print("\nResumen original:")
        print(summary[["mean±std"]].to_string())
        print("\nPrimeras filas:")
        print(df_inst.head(10).to_string(index=False))

        original_csv = (
            OUTPUT_DIR
            / f"vehicle_shap_lcd_pair_metrics_{pair_tag}.csv"
        )
        df_inst.to_csv(original_csv, index=False)

        print("\n--- Comparación completa de consistencia ---")

        df_comparison, summary_comparison = (
            explainer.evaluate_shap_lcd_pair_compare_consistency(
                pair=pair,
                class_names=CLASS_NAMES,
                filter_mode=FILTER_MODE,
                n_instances=N_INSTANCES,
                M=M,
                sigma=SIGMA,
                top_k=TOP_K,
                tau=TAU,
                normalize=NORMALIZE,
                seed=SEED
            )
        )

        print("\nResumen completo:")
        print(summary_comparison[["mean±std"]].to_string())
        print("\nPrimeras filas:")
        print(df_comparison.head(10).to_string(index=False))

        comparison_csv = (
            OUTPUT_DIR
            / f"vehicle_compare_consistency_{pair_tag}.csv"
        )
        df_comparison.to_csv(comparison_csv, index=False)

        evaluated_indices = df_inst["idx_test"].to_numpy()
        print("\ny_test counts:")
        print(
            pd.Series(
                explainer.y_test[evaluated_indices]
            ).value_counts().sort_index()
        )

        print_statistical_summary(
            df_inst,
            m_value=M,
            pair_name=pair_name
        )

        df_inst_with_pair = df_inst.copy()
        df_inst_with_pair["pair_code"] = pair_tag
        all_original_results.append(df_inst_with_pair)

        df_comparison_with_pair = df_comparison.copy()
        df_comparison_with_pair["pair_code"] = pair_tag
        all_comparison_results.append(df_comparison_with_pair)

        final_summary_rows.append({
            "pair": pair_name,
            "pair_code": pair_tag,
            "N": len(df_comparison),
            "Stability_S_mean": df_comparison["Stability_S"].mean(),
            "Stability_S_std": df_comparison["Stability_S"].std(ddof=1),
            "Consistency_C_predexp_mean": (
                df_comparison["Consistency_C_predexp"].mean()
            ),
            "Consistency_C_predexp_std": (
                df_comparison["Consistency_C_predexp"].std(ddof=1)
            ),
            "Consistency_C_imp_mean": (
                df_comparison["Consistency_C_imp"].mean()
            ),
            "Consistency_C_imp_std": (
                df_comparison["Consistency_C_imp"].std(ddof=1)
            ),
            "Coverage_imp_mean": (
                df_comparison["Coverage_imp"].mean()
            ),
            "PredKeep_imp_mean": (
                df_comparison["PredKeep_imp"].mean()
            ),
        })

    all_original_df = pd.concat(
        all_original_results,
        ignore_index=True
    )
    all_comparison_df = pd.concat(
        all_comparison_results,
        ignore_index=True
    )
    final_summary_df = pd.DataFrame(final_summary_rows)

    all_original_df.to_csv(
        OUTPUT_DIR / "vehicle_all_pairs_original_metrics.csv",
        index=False
    )
    all_comparison_df.to_csv(
        OUTPUT_DIR / "vehicle_all_pairs_comparison_metrics.csv",
        index=False
    )
    final_summary_df.to_csv(
        OUTPUT_DIR / "vehicle_metrics_summary.csv",
        index=False
    )

    print("\n" + "=" * 90)
    print(" RESUMEN FINAL DE MÉTRICAS - VEHICLE")
    print("=" * 90)
    print(final_summary_df.to_string(index=False))
    print("=" * 90)

    print("\n" + "=" * 72)
    print(" EJECUCIÓN DE VEHICLE FINALIZADA")
    print("=" * 72)
    print("Resultados guardados en:")
    print(OUTPUT_DIR)
    print("=" * 72)
