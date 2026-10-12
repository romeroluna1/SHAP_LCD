from pathlib import Path
import os
import random
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd import SHAPExplainer


# ============================================================
# SEMILLA
# ============================================================

SEED = 42

# True ejecuta una prueba rápida de integración.
# False reproduce la configuración completa del artículo.
QUICK_TEST = False
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# PARÁMETROS GENERALES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "wine"
RESULTS_ROOT = PROJECT_ROOT / "results" / "wine"
RESULTS_DIR = RESULTS_ROOT / ("quick_test" if QUICK_TEST else "final")

CSV_PATH = DATA_DIR / "wine.csv"
TARGET_COLUMN = "target"

# Instancia utilizada en el artículo publicado
INSTANCE_INDEX = 38

# IMPORTANTE:
# El pickle antiguo:
#
#     shap_values_wine_xgb.pickle
#
# contiene SHAP en escala raw/margin.
#
# NO debe reutilizarse en este experimento.
SHAP_FILE = (
    ARTIFACT_DIR
    / "shap_values_wine_xgb_probability.pickle"
)

# El pickle de probabilidad ya fue recalculado y verificado
# en dos ejecuciones independientes.
#
# "auto"   -> carga si existe; si no, calcula.
# "always" -> recalcula siempre.
SHAP_MODE = "auto"


# Nueva instancia externa
NEW_INSTANCE = [
    13.2,
    2.7,
    2.4,
    17.5,
    100.0,
    2.8,
    2.9,
    0.30,
    2.0,
    5.2,
    1.05,
    3.3,
    1050.0
]


# ============================================================
# PARÁMETROS DE ESTABILIDAD Y CONSISTENCIA
# ============================================================

M_PERTURB = 2 if QUICK_TEST else 40
N_INSTANCES = 2 if QUICK_TEST else None
SIGMA = 0.02
TOP_K = 10
TAU = 0.90
NORMALIZE = "l2"

# Pares de Wine utilizados para la auditoría del artículo.
#
# (0,1) = wine_class_0 vs wine_class_1
# (0,2) = wine_class_0 vs wine_class_2
METRIC_PAIRS = [
    (0, 1),
    (0, 2)
]


# ============================================================
# NOMBRES DE CLASE
# ============================================================

df = pd.read_csv(CSV_PATH)

CLASS_NAMES = {
    0: "wine_class_0",
    1: "wine_class_1",
    2: "wine_class_2",
}


# ============================================================
# FUNCIÓN AUXILIAR:
# EXTRAER SHAP DE UNA INSTANCIA PARA CADA CLASE
# ============================================================

def get_shap_instance(shap_values, instance_index):
    """
    Devuelve:

        [SHAP_clase_0,
         SHAP_clase_1,
         SHAP_clase_2]

    Compatible con versiones de SHAP que devuelven:

        - lista por clase;
        - ndarray (n_instancias, n_features, n_clases).
    """

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


# ============================================================
# VERIFICACIÓN DE ESCALA SHAP
# ============================================================

def verify_probability_scale(explainer, instance_index):
    """
    Comprueba para cada clase:

        expected_value + sum(SHAP)
        ≈
        predict_proba
    """

    print("\n")
    print("=" * 70)
    print(" VERIFICACIÓN DE LA ESCALA SHAP")
    print("=" * 70)

    x = np.asarray(
        explainer.X_test[instance_index]
    ).reshape(1, -1)

    probabilities = (
        explainer.modelo_entrenado_70
        .predict_proba(x)[0]
    )

    shap_instance = get_shap_instance(
        explainer.shap_values,
        instance_index
    )

    expected_value = np.asarray(
        explainer.explainer.expected_value
    ).reshape(-1)

    print(
        "\n"
        "Clase               "
        "predict_proba       "
        "base + sum(SHAP)     "
        "error"
    )

    print("-" * 70)

    errors = []

    for c in range(len(probabilities)):

        reconstructed = (
            expected_value[c]
            +
            np.sum(shap_instance[c])
        )

        error = abs(
            probabilities[c] - reconstructed
        )

        errors.append(error)

        print(
            f"{CLASS_NAMES[c]:18s} "
            f"{probabilities[c]:.10f}      "
            f"{reconstructed:.10f}      "
            f"{error:.3e}"
        )

    print("-" * 70)

    max_error = max(errors)

    print(
        f"\nError máximo de reconstrucción: "
        f"{max_error:.3e}"
    )

    if max_error < 1e-5:

        print(
            "\n✅ VERIFICACIÓN SUPERADA:"
        )

        print(
            "Los valores SHAP están explicando predict_proba."
        )

        print(
            "Por tanto, SHAP-LCD está trabajando "
            "en escala de probabilidad."
        )

    else:

        print(
            "\n⚠️ ADVERTENCIA:"
        )

        print(
            "La reconstrucción no coincide suficientemente "
            "con predict_proba."
        )

    print("=" * 70)


# ============================================================
# MOSTRAR DELTA S DE UNA INSTANCIA
# ============================================================

def print_pair_shap_differences(
    explainer,
    instance_index,
    class_i,
    class_j
):
    """
    Muestra:

        ΔS = SHAP(C_i) - SHAP(C_j)

    ordenado por |ΔS|.
    """

    shap_instance = get_shap_instance(
        explainer.shap_values,
        instance_index
    )

    delta_s = (
        shap_instance[class_i]
        -
        shap_instance[class_j]
    )

    feature_names = (
        explainer.df
        .drop(columns=[TARGET_COLUMN])
        .columns
        .tolist()
    )

    order = np.argsort(
        np.abs(delta_s)
    )[::-1]

    print("\n")
    print("=" * 75)

    print(
        f" ΔS EN PROBABILIDAD: "
        f"{CLASS_NAMES[class_i]} "
        f"vs "
        f"{CLASS_NAMES[class_j]}"
    )

    print("=" * 75)

    print(
        f"{'Característica':30s} "
        f"{'Delta S':>15s} "
        f"{'Dirección':>20s}"
    )

    print("-" * 75)

    for idx in order:

        value = delta_s[idx]

        if value > 0:

            direction = CLASS_NAMES[class_i]

        elif value < 0:

            direction = CLASS_NAMES[class_j]

        else:

            direction = "Neutral"

        print(
            f"{feature_names[idx]:30s} "
            f"{value:15.8f} "
            f"{direction:>20s}"
        )

    print("=" * 75)


# ============================================================
# ESTADÍSTICOS COMPLEMENTARIOS PARA N
# ============================================================

def print_statistical_details(
    df_inst,
    pair,
    M
):
    """
    Reproduce los estadísticos que estaban al final
    del main original para Stability y Consistency.
    """

    N = len(df_inst)

    mean_S = float(
        df_inst["Stability_S"].mean()
    )

    mean_C = float(
        df_inst["Consistency_C"].mean()
    )

    s_S = float(
        df_inst["Stability_S"].std(ddof=1)
    )

    s_C = float(
        df_inst["Consistency_C"].std(ddof=1)
    )

    SE_S = (
        s_S / np.sqrt(N)
        if N > 0
        else np.nan
    )

    SE_C = (
        s_C / np.sqrt(N)
        if N > 0
        else np.nan
    )

    z = 1.96

    ciS_low = mean_S - z * SE_S
    ciS_high = mean_S + z * SE_S

    ciC_low = mean_C - z * SE_C
    ciC_high = mean_C + z * SE_C

    print("\n")
    print("=" * 70)
    print(
        f" ESTADÍSTICOS PARA JUSTIFICAR N "
        f"- PAR {pair[0]} vs {pair[1]}"
    )
    print("=" * 70)

    print(
        f"N instancias evaluadas: {N}"
    )

    print(
        f"M perturbaciones por instancia: {M}"
    )

    print(
        "\n--- Promedios globales ---"
    )

    print(
        f"Stability_S mean: {mean_S:.6f}"
    )

    print(
        f"Consistency_C mean: {mean_C:.6f}"
    )

    print(
        "\n--- Dispersión entre instancias ---"
    )

    print(
        f"s_S (std Stability_S): {s_S:.6f}"
    )

    print(
        f"s_C (std Consistency_C): {s_C:.6f}"
    )

    print(
        "\n--- Error estándar del promedio ---"
    )

    print(
        f"SE(S_mean): {SE_S:.6f}"
    )

    print(
        f"SE(C_mean): {SE_C:.6f}"
    )

    print(
        "\n--- Intervalo de confianza aproximado 95% ---"
    )

    print(
        f"CI95(S_mean): "
        f"[{ciS_low:.6f}, {ciS_high:.6f}]"
    )

    print(
        f"CI95(C_mean): "
        f"[{ciC_low:.6f}, {ciC_high:.6f}]"
    )

    print("=" * 70)


# ============================================================
# MÉTRICAS PARA UN PAR
# ============================================================

def evaluate_probability_metrics(
    explainer,
    pair
):
    """
    Calcula en UNA SOLA PASADA:

        Stability_S
        Consistency_C_predexp
        Consistency_C_imp
        Coverage_imp
        PredKeep_imp

    La columna Consistency_C_predexp corresponde a
    Consistency_C del evaluate_shap_lcd_pair original.

    Se evita ejecutar dos veces miles de explicaciones
    Kernel SHAP.
    """

    print("\n")
    print("#" * 75)

    print(
        f" MÉTRICAS SHAP-LCD EN PROBABILIDAD: "
        f"{CLASS_NAMES[pair[0]]} vs "
        f"{CLASS_NAMES[pair[1]]}"
    )

    print("#" * 75)

    print(
        f"M       = {M_PERTURB}"
    )

    print(
        f"sigma   = {SIGMA}"
    )

    print(
        f"Top-K   = {TOP_K}"
    )

    print(
        f"tau     = {TAU}"
    )

    print(
        f"norma   = {NORMALIZE}"
    )

    print(
        f"seed    = {SEED}"
    )

    print(
        "filtro  = etiqueta verdadera"
    )

    print(
        f"n_instances = {N_INSTANCES}"
        if N_INSTANCES is not None
        else "n_instances = todas las disponibles"
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    # reiniciamos las semillas justo antes de las métricas.
    #
    # De esta forma el bloque de métricas no depende de
    # operaciones aleatorias que hayan ocurrido anteriormente.
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)

    # --------------------------------------------------------
    # UNA SOLA EVALUACIÓN
    # --------------------------------------------------------

    df_cmp, summary_cmp = (
        explainer
        .evaluate_shap_lcd_pair_compare_consistency(
            pair=pair,
            class_names=CLASS_NAMES,
            filter_mode="true",
            n_instances=N_INSTANCES,
            M=M_PERTURB,
            sigma=SIGMA,
            top_k=TOP_K,
            tau=TAU,
            normalize=NORMALIZE,
            seed=SEED
        )
    )

    # --------------------------------------------------------
    # RECONSTRUIR LA SALIDA DEL evaluate_shap_lcd_pair
    # ORIGINAL SIN RECALCULAR SHAP
    # --------------------------------------------------------

    df_basic = df_cmp[
        [
            "idx_test",
            "pair",
            "abs_deltaP_ij",
            "Stability_S",
            "Consistency_C_predexp"
        ]
    ].copy()

    df_basic = df_basic.rename(
        columns={
            "Consistency_C_predexp":
                "Consistency_C"
        }
    )

    summary_basic = (
        df_basic[
            [
                "Stability_S",
                "Consistency_C"
            ]
        ]
        .agg(["mean", "std"])
        .T
    )

    summary_basic["mean±std"] = (
        summary_basic["mean"]
        .map(lambda x: f"{x:.6f}")
        +
        " ± "
        +
        summary_basic["std"]
        .map(lambda x: f"{x:.6f}")
    )

    # --------------------------------------------------------
    # RESULTADOS TIPO MAIN ORIGINAL
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)

    print(
        " RESUMEN ORIGINAL: "
        "STABILITY + CONSISTENCY pred->exp"
    )

    print("-" * 70)

    print(
        summary_basic[
            ["mean±std"]
        ].to_string()
    )

    print(
        "\nPrimeras filas:"
    )

    print(
        df_basic
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # COMPARACIÓN DE LAS DOS DEFINICIONES DE CONSISTENCIA
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)

    print(
        " COMPARACIÓN COMPLETA DE CONSISTENCIA"
    )

    print("-" * 70)

    print(
        summary_cmp[
            ["mean±std"]
        ].to_string()
    )

    print(
        "\nPrimeras filas:"
    )

    print(
        df_cmp
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # DISTRIBUCIÓN DE CLASES REALES
    # --------------------------------------------------------

    idxs = (
        df_cmp["idx_test"]
        .to_numpy(dtype=int)
    )

    y_eval = np.asarray(
        explainer.y_test
    )[idxs]

    print(
        "\ny_test counts en las instancias evaluadas:"
    )

    print(
        pd.Series(y_eval)
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # GUARDAR CSV
    # NO SOBREESCRIBIR LOS ARCHIVOS RAW ORIGINALES
    # --------------------------------------------------------

    pair_tag = (
        f"{pair[0]}_{pair[1]}"
    )

    basic_file = (
        f"wine_shap_lcd_probability_"
        f"pair_metrics_{pair_tag}.csv"
    )

    compare_file = (
        f"wine_probability_"
        f"compare_consistency_{pair_tag}.csv"
    )

    df_basic.to_csv(
        basic_file,
        index=False
    )

    df_cmp.to_csv(
        compare_file,
        index=False
    )

    print(
        f"\nCSV guardado: {basic_file}"
    )

    print(
        f"CSV guardado: {compare_file}"
    )

    # --------------------------------------------------------
    # ESTADÍSTICOS PARA N
    # --------------------------------------------------------

    print_statistical_details(
        df_basic,
        pair,
        M_PERTURB
    )

    # --------------------------------------------------------
    # DEVOLVER UNA FILA RESUMEN PARA COMPARAR PARES
    # --------------------------------------------------------

    result_row = {

        "pair":
            f"{CLASS_NAMES[pair[0]]} vs "
            f"{CLASS_NAMES[pair[1]]}",

        "N":
            len(df_cmp),

        "Stability_S_mean":
            float(
                df_cmp[
                    "Stability_S"
                ].mean()
            ),

        "Stability_S_std":
            float(
                df_cmp[
                    "Stability_S"
                ].std(ddof=1)
            ),

        "Consistency_C_predexp_mean":
            float(
                df_cmp[
                    "Consistency_C_predexp"
                ].mean()
            ),

        "Consistency_C_predexp_std":
            float(
                df_cmp[
                    "Consistency_C_predexp"
                ].std(ddof=1)
            ),

        "Consistency_C_imp_mean":
            float(
                df_cmp[
                    "Consistency_C_imp"
                ].mean()
            ),

        "Consistency_C_imp_std":
            float(
                df_cmp[
                    "Consistency_C_imp"
                ].std(ddof=1)
            ),

        "Coverage_imp_mean":
            float(
                df_cmp[
                    "Coverage_imp"
                ].mean()
            ),

        "PredKeep_imp_mean":
            float(
                df_cmp[
                    "PredKeep_imp"
                ].mean()
            )
    }

    return (
        df_basic,
        df_cmp,
        result_row
    )


# ============================================================
# BLOQUE PRINCIPAL
# ============================================================

if __name__ == "__main__":

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Las salidas relativas del experimento se almacenan aquí.
    os.chdir(RESULTS_DIR)

    print("\n" + "=" * 70)
    print(" SHAP-LCD - WINE")
    print("=" * 70)
    print(f"Dataset      : {CSV_PATH}")
    print(f"SHAP pickle  : {SHAP_FILE}")
    print(f"Resultados   : {RESULTS_DIR}")
    print(f"Quick test   : {QUICK_TEST}")
    print(f"M            : {M_PERTURB}")
    print(f"N instances  : {N_INSTANCES}")
    print("=" * 70)

    # ========================================================
    # 1. CREAR SHAPEXPLAINER
    # ========================================================

    explainer = SHAPExplainer(
        csv_path=CSV_PATH,
        model_class=lambda **kwargs: XGBClassifier(
            random_state=SEED,
            eval_metric="mlogloss",
        ),
        target_column=TARGET_COLUMN,
        random_state=SEED,
        output_scale="probability",
        background_size=100,
    )


    # ========================================================
    # 2. ENTRENAR EXACTAMENTE EL MISMO XGBOOST
    # ========================================================

    explainer.train_and_evaluate()

    # ========================================================
    # 3. CALCULAR / CARGAR SHAP EN PROBABILIDAD
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" SHAP EN ESCALA DE PROBABILIDAD")
    print("=" * 70)

    print(
        "Archivo:",
        SHAP_FILE
    )

    np.random.seed(SEED)

    explainer.ensure_shap_values(
        SHAP_FILE,
        mode=SHAP_MODE
    )

    print(
        "\nX_test shape:",
        explainer.X_test.shape
    )

    print(
        "SHAP values shape:",
        np.asarray(
            explainer.shap_values
        ).shape
    )


    # ========================================================
    # 6. VERIFICAR MATEMÁTICAMENTE LA ESCALA
    # ========================================================

    verify_probability_scale(
        explainer,
        INSTANCE_INDEX
    )


    # ========================================================
    # 7. PROBABILIDADES INSTANCIA 38
    # ========================================================

    x_instance = np.asarray(
        explainer.X_test[
            INSTANCE_INDEX
        ]
    ).reshape(1, -1)

    probabilities = (
        explainer.modelo_entrenado_70
        .predict_proba(
            x_instance
        )[0]
    )

    print("\n")
    print("=" * 70)
    print(" PROBABILIDADES - INSTANCIA 38")
    print("=" * 70)

    for c, p in enumerate(
        probabilities
    ):

        print(
            f"{CLASS_NAMES[c]:15s}: "
            f"{p:.10f}"
        )

    print("=" * 70)


    # ========================================================
    # 8. ALGORITMO 1
    # ========================================================

    print(
        "\n--- Algoritmo 1: "
        "Diferencias de probabilidad entre clases ---"
    )

    prob_diffs = (
        explainer
        .plot_probability_differences(

            INSTANCE_INDEX,

            instance_type="test",

            class_names=CLASS_NAMES,

            instance_label=(
                f"Instancia "
                f"{INSTANCE_INDEX}"
            ),

            save_path=(
                RESULTS_DIR
                / f"probability_differences_instance_{INSTANCE_INDEX}.png"
            ),

            dpi=600
        )
    )

    for i, (
        diff,
        c1,
        c2
    ) in enumerate(prob_diffs):

        print(
            f"{i + 1}. "
            f"{c1} vs {c2}: "
            f"|Delta P| = "
            f"{diff:.8f}"
        )


    # ========================================================
    # 9. MOSTRAR LOS TRES PARES
    # ========================================================

    print_pair_shap_differences(
        explainer,
        INSTANCE_INDEX,
        0,
        1
    )

    print_pair_shap_differences(
        explainer,
        INSTANCE_INDEX,
        1,
        2
    )

    print_pair_shap_differences(
        explainer,
        INSTANCE_INDEX,
        0,
        2
    )


    # ========================================================
    # 10. ALGORITMO 2
    # ========================================================

    print(
        "\nIngrese los números de los pares "
        "que desea analizar "
        "(ej. 1 o 1,2):"
    )

    user_input = input(
        "Opción(es): "
    )

    selected_indices = [

        int(i.strip()) - 1

        for i in user_input.split(",")

        if i.strip().isdigit()
    ]

    selected_pairs = [

        (
            prob_diffs[i][1],
            prob_diffs[i][2]
        )

        for i in selected_indices

        if 0 <= i < len(
            prob_diffs
        )
    ]

    print(
        "\n--- Algoritmo 2: "
        "Diferencias SHAP "
        "en probabilidad ---"
    )

    explainer.explain_selected_pairs(

        INSTANCE_INDEX,

        selected_pairs=(
            selected_pairs
        ),

        instance_type="test",

        class_names=CLASS_NAMES,

        instance_label=(
            f"Instancia "
            f"{INSTANCE_INDEX}"
        ),

        save_dir=RESULTS_DIR,

        dpi=600
    )


    # ========================================================
    # 11. NUEVA INSTANCIA EXTERNA
    # ========================================================

    print(
        "\n--- Algoritmo 1: "
        "Diferencias de probabilidad "
        "para instancia externa ---"
    )

    new_diffs = (
        explainer
        .plot_probability_differences(

            NEW_INSTANCE,

            instance_type="new",

            class_names=CLASS_NAMES,

            instance_label=(
                "Instancia nueva"
            ),

            save_path=(
                RESULTS_DIR
                / "probability_differences_external_instance.png"
            ),

            dpi=600
        )
    )

    for i, (
        diff,
        c1,
        c2
    ) in enumerate(new_diffs):

        print(
            f"{i + 1}. "
            f"{c1} vs {c2}: "
            f"|Delta P| = "
            f"{diff:.8f}"
        )

    print(
        "\nIngrese los números de los pares "
        "que desea analizar para "
        "la instancia externa "
        "(ej. 1 o 1,2):"
    )

    new_input = input(
        "Opción(es): "
    )

    selected_indices_ext = [

        int(i.strip()) - 1

        for i in new_input.split(",")

        if i.strip().isdigit()
    ]

    selected_new_pairs = [

        (
            new_diffs[i][1],
            new_diffs[i][2]
        )

        for i in selected_indices_ext

        if 0 <= i < len(
            new_diffs
        )
    ]

    print(
        "\n--- Algoritmo 2: "
        "Diferencias SHAP "
        "en probabilidad "
        "para instancia externa ---"
    )

    explainer.explain_selected_pairs(

        NEW_INSTANCE,

        selected_pairs=(
            selected_new_pairs
        ),

        instance_type="new",

        class_names=CLASS_NAMES,

        instance_label=(
            "Instancia nueva"
        ),

        save_dir=RESULTS_DIR,

        dpi=600
    )


    # ========================================================
    # 12. FIGURAS GLOBALES
    # ========================================================

    explainer.plot_global_violin_by_class(

        class_names=CLASS_NAMES,

        use_normalized=True,

        mode="pooled_features",

        dataset_name="Wine",

        save_path=(
            "wine_global_violin_probability.png"
        ),

        dpi=600,

        figsize=(8, 5),

        violin_alpha=0.75,

        fliersize=8
    )


    explainer.plot_class_distribution_bar(

        class_names=CLASS_NAMES,

        dataset_name="Wine",

        save_path=(
            "wine_class_distribution.png"
        ),

        dpi=600,

        figsize=(8, 5),

        cmap_min=0.2,

        cmap_max=0.8,

        show_values=True
    )


    # ========================================================
    # 13. MÉTRICAS SHAP-LCD EN PROBABILIDAD
    # ========================================================

    print("\n")
    print("=" * 75)
    print(" MÉTRICAS SHAP-LCD EN ESCALA DE PROBABILIDAD")
    print("=" * 75)

    all_pair_summaries = []

    for pair in METRIC_PAIRS:

        (
            df_basic,
            df_cmp,
            result_row
        ) = evaluate_probability_metrics(
            explainer,
            pair
        )

        all_pair_summaries.append(
            result_row
        )


    # ========================================================
    # 14. RESUMEN DE LOS DOS PARES
    # ========================================================

    df_pair_summary = pd.DataFrame(
        all_pair_summaries
    )

    print("\n")
    print("=" * 90)
    print(" RESUMEN FINAL MÉTRICAS WINE EN PROBABILIDAD")
    print("=" * 90)

    print(
        df_pair_summary
        .to_string(index=False)
    )

    df_pair_summary.to_csv(
        "wine_probability_metrics_summary.csv",
        index=False
    )

    print(
        "\nResumen guardado en:"
    )

    print(
        "wine_probability_metrics_summary.csv"
    )


    # ========================================================
    # 15. INFORMACIÓN FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" EXPERIMENTO WINE ACTUALIZADO")
    print("=" * 70)

    print(
        "Modelo:",
        explainer.modelo_entrenado_70
        .__class__.__name__
    )

    print(
        "Función explicada por SHAP: "
        "predict_proba"
    )

    print(
        "Escala SHAP: probabilidad"
    )

    print(
        "Semilla:",
        SEED
    )

    print(
        "Instancia principal:",
        INSTANCE_INDEX
    )

    print(
        "Archivo SHAP:",
        SHAP_FILE
    )

    print(
        "Pares de métricas:",
        METRIC_PAIRS
    )

    print(
        "M:",
        M_PERTURB
    )

    print(
        "sigma:",
        SIGMA
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "tau:",
        TAU
    )

    print(
        "normalización:",
        NORMALIZE
    )

    print("=" * 70)
