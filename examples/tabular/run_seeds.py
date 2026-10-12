"""Ejemplo local reproducible de SHAP-LCD para Seeds."""

from __future__ import annotations

import os
import pickle
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.neural_network import MLPClassifier


# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd import SHAPExplainer  # noqa: E402


# =============================================================================
# CONFIGURACION REPRODUCIBLE
# =============================================================================

SEED = 42
TARGET_COLUMN = "target"
SHOW_PLOTS = True

CLASS_NAMES = {
    0: "Kama",
    1: "Rosa",
    2: "Canadian",
}

# Par utilizado en el ejemplo global de Seeds.
LOCAL_PAIR = (0, 1)

CSV_PATH = PROJECT_ROOT / "data" / "seeds.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "seeds"

# Pickle proveniente de SHAP-GLCD. Solo se reutiliza si supera todas las
# verificaciones de compatibilidad y reconstruccion probabilistica.
GLCD_SHAP_FILE = (
    ARTIFACTS_DIR
    / "shap_values_seeds_mlp_probability_0_vs_1.pickle"
)

# Cache propia de SHAP-LCD. Se crea automaticamente si el pickle anterior no
# resulta compatible con el modelo entrenado en el entorno actual.
LCD_SHAP_FILE = (
    ARTIFACTS_DIR
    / "shap_values_seeds_lcd_mlp_probability.pickle"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "seeds" / "local"


def build_model(random_state: int = SEED) -> MLPClassifier:
    """Construye exactamente el MLP utilizado en Seeds SHAP-GLCD."""
    return MLPClassifier(
        hidden_layer_sizes=(50, 30),
        max_iter=2000,
        random_state=random_state,
    )


def print_environment() -> None:
    print("Python          :", sys.version)
    print("NumPy          :", np.__version__)
    print("pandas         :", pd.__version__)
    print("scikit-learn   :", sklearn.__version__)
    print("SHAP           :", shap.__version__)
    print("matplotlib     :", matplotlib.__version__)
    print("seaborn        :", sns.__version__)
    print("OS             :", platform.platform())
    print("Processor      :", platform.processor())


def expected_values(explainer: SHAPExplainer) -> np.ndarray:
    values = getattr(explainer.explainer, "expected_value", None)
    if values is None:
        raise RuntimeError("El explicador SHAP no contiene expected_value.")
    return np.asarray(values, dtype=float).reshape(-1)


def reconstruction_error(
    explainer: SHAPExplainer,
    shap_values: np.ndarray,
    base_values: np.ndarray,
) -> tuple[float, float]:
    shap_array = np.asarray(shap_values, dtype=float)
    probabilities = explainer.modelo_entrenado_70.predict_proba(
        explainer.X_test
    )

    if shap_array.shape != (
        len(explainer.X_test),
        explainer.X_test.shape[1],
        probabilities.shape[1],
    ):
        raise ValueError(
            "Forma SHAP incompatible: "
            f"{shap_array.shape}; se esperaba "
            f"{(len(explainer.X_test), explainer.X_test.shape[1], probabilities.shape[1])}."
        )

    reconstructed = base_values.reshape(1, -1) + shap_array.sum(axis=1)
    errors = np.abs(probabilities - reconstructed)
    return float(errors.mean()), float(errors.max())


def validate_glcd_metadata(payload: dict, explainer: SHAPExplainer) -> None:
    required = {
        "shap_values",
        "expected_value",
        "output_space",
        "random_state",
        "feature_names",
        "class_names",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Faltan claves en el pickle: {sorted(missing)}")

    feature_names = explainer.df.drop(columns=[TARGET_COLUMN]).columns.tolist()
    if list(payload["feature_names"]) != feature_names:
        raise ValueError("Los nombres u orden de las caracteristicas no coinciden.")

    if [int(value) for value in payload["class_names"]] != list(CLASS_NAMES):
        raise ValueError("El orden de las clases del pickle no coincide.")

    if int(payload["random_state"]) != SEED:
        raise ValueError("La semilla del pickle no coincide.")

    if str(payload["output_space"]).lower() != "probability":
        raise ValueError("El pickle no contiene SHAP en escala de probabilidad.")


def calculate_and_save_current_shap(explainer: SHAPExplainer) -> None:
    print("Calculando una cache SHAP-LCD compatible con el modelo actual...")
    explainer.calculate_shap_values()
    explainer.save_shap_values(LCD_SHAP_FILE)


def load_or_create_shap(explainer: SHAPExplainer) -> None:
    """Carga una cache valida o calcula una nueva sin sobrescribir SHAP-GLCD."""
    tolerance = 1e-5

    if LCD_SHAP_FILE.exists():
        print(f"\nCargando cache SHAP-LCD:\n{LCD_SHAP_FILE}")
        explainer.load_shap_values(LCD_SHAP_FILE)
        try:
            mean_error, max_error = reconstruction_error(
                explainer,
                explainer.shap_values,
                expected_values(explainer),
            )
        except Exception as error:
            print(f"Cache SHAP-LCD incompatible: {error}")
            calculate_and_save_current_shap(explainer)
        else:
            if max_error >= tolerance:
                print(
                    "Cache SHAP-LCD desactualizada: "
                    f"error maximo={max_error:.3e}."
                )
                calculate_and_save_current_shap(explainer)
            else:
                print(
                    "Cache SHAP-LCD verificada: "
                    f"error medio={mean_error:.3e}, "
                    f"error maximo={max_error:.3e}."
                )
        return

    if GLCD_SHAP_FILE.exists():
        print(f"\nVerificando pickle SHAP-GLCD:\n{GLCD_SHAP_FILE}")
        try:
            with GLCD_SHAP_FILE.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                raise TypeError("El objeto principal no es un diccionario.")

            validate_glcd_metadata(payload, explainer)
            shap_values = np.asarray(payload["shap_values"], dtype=float)
            stored_base = np.asarray(payload["expected_value"], dtype=float)

            stored_mean, stored_max = reconstruction_error(
                explainer,
                shap_values,
                stored_base,
            )
            current_mean, current_max = reconstruction_error(
                explainer,
                shap_values,
                expected_values(explainer),
            )

            print(
                "Error con base almacenada -> "
                f"medio={stored_mean:.3e}, maximo={stored_max:.3e}"
            )
            print(
                "Error con base actual     -> "
                f"medio={current_mean:.3e}, maximo={current_max:.3e}"
            )

            if stored_max >= tolerance or current_max >= tolerance:
                raise ValueError(
                    "el pickle no reconstruye predict_proba con el modelo actual"
                )

        except Exception as error:
            print(f"No se reutilizara el pickle SHAP-GLCD: {error}.")
            calculate_and_save_current_shap(explainer)
        else:
            explainer.shap_values = shap_values
            explainer.save_shap_values(LCD_SHAP_FILE)
            print("Pickle SHAP-GLCD validado y convertido a cache SHAP-LCD.")
        return

    print("\nNo existe ninguna cache SHAP para Seeds.")
    calculate_and_save_current_shap(explainer)


def verify_probability_scale(explainer: SHAPExplainer) -> None:
    mean_error, max_error = reconstruction_error(
        explainer,
        explainer.shap_values,
        expected_values(explainer),
    )

    print("\n" + "=" * 72)
    print(" VERIFICACION DE LA ESCALA SHAP - SEEDS")
    print("=" * 72)
    print("Salida explicada : predict_proba")
    print(f"Error medio      : {mean_error:.3e}")
    print(f"Error maximo     : {max_error:.3e}")

    if max_error >= 1e-5:
        raise RuntimeError(
            "La cache SHAP no reconstruye predict_proba. "
            "No se generaran figuras SHAP-LCD."
        )

    print("VERIFICACION SUPERADA: SHAP esta en probabilidad.")
    print("=" * 72)


def select_local_instance(explainer: SHAPExplainer) -> int:
    """Selecciona reproduciblemente una instancia correcta e informativa."""
    class_j, class_r = LOCAL_PAIR
    predictions = explainer.modelo_entrenado_70.predict(explainer.X_test)
    probabilities = explainer.modelo_entrenado_70.predict_proba(explainer.X_test)

    candidates = np.where(
        np.isin(explainer.y_test, [class_j, class_r])
        & (predictions == explainer.y_test)
    )[0]
    if len(candidates) == 0:
        candidates = np.where(
            np.isin(explainer.y_test, [class_j, class_r])
        )[0]

    # Favorece una instancia en la que ambas clases del par tengan presencia.
    pair_support = np.minimum(
        probabilities[candidates, class_j],
        probabilities[candidates, class_r],
    )
    return int(candidates[np.argmax(pair_support)])


def save_selected_instance(
    explainer: SHAPExplainer,
    instance_index: int,
) -> None:
    raw_values = explainer.X[instance_index]
    # X_test no conserva indices del dataset original. Los valores sin escalar
    # se recuperan exactamente mediante el scaler ajustado.
    raw_values = explainer.scaler.inverse_transform(
        explainer.X_test[[instance_index]]
    )[0]
    probabilities = explainer.modelo_entrenado_70.predict_proba(
        explainer.X_test[[instance_index]]
    )[0]
    prediction = int(np.argmax(probabilities))
    true_label = int(explainer.y_test[instance_index])

    row = {
        "test_position": instance_index,
        "true_class_index": true_label,
        "true_class": CLASS_NAMES[true_label],
        "predicted_class_index": prediction,
        "predicted_class": CLASS_NAMES[prediction],
    }
    for name, value in zip(
        explainer.df.drop(columns=[TARGET_COLUMN]).columns,
        raw_values,
    ):
        row[name] = float(value)
    for class_index, probability in enumerate(probabilities):
        row[f"P_{CLASS_NAMES[class_index]}"] = float(probability)

    pd.DataFrame([row]).to_csv(
        RESULTS_DIR / "selected_instance.csv",
        index=False,
    )

    print("\nSelected local instance")
    print(f"Test position   : {instance_index}")
    print(f"True class      : {CLASS_NAMES[true_label]}")
    print(f"Predicted class : {CLASS_NAMES[prediction]}")
    for class_index, probability in enumerate(probabilities):
        print(f"P({CLASS_NAMES[class_index]}) = {probability:.8f}")


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el dataset:\n{CSV_PATH}")

    print_environment()
    print("\n" + "=" * 72)
    print(" SHAP-LCD - SEEDS - LOCAL TABULAR EXAMPLE")
    print("=" * 72)
    print(f"Dataset         : {CSV_PATH}")
    print(f"GLCD pickle     : {GLCD_SHAP_FILE}")
    print(f"LCD cache       : {LCD_SHAP_FILE}")
    print(f"Fixed pair      : {CLASS_NAMES[0]} vs {CLASS_NAMES[1]}")
    print(f"Results         : {RESULTS_DIR}")
    print(f"Show plots      : {SHOW_PLOTS}")
    print("=" * 72)

    explainer = SHAPExplainer(
        csv_path=CSV_PATH,
        model_class=build_model,
        target_column=TARGET_COLUMN,
        random_state=SEED,
        output_scale="probability",
        background_size=100,
    )
    explainer.train_and_evaluate()
    load_or_create_shap(explainer)
    verify_probability_scale(explainer)

    instance_index = select_local_instance(explainer)
    save_selected_instance(explainer, instance_index)

    print("\nAlgorithm 1 - probability differences |Delta P|")
    ranking = explainer.plot_probability_differences(
        instance_index,
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Seeds instance {instance_index}",
        save_path=RESULTS_DIR / "probability_differences.png",
        dpi=600,
        show=SHOW_PLOTS,
    )
    for position, (difference, class_i, class_j) in enumerate(ranking, start=1):
        print(
            f"{position}. {class_i} vs {class_j}: "
            f"|Delta P|={difference:.8f}"
        )

    fixed_names = (CLASS_NAMES[LOCAL_PAIR[0]], CLASS_NAMES[LOCAL_PAIR[1]])
    ranking_pairs = [(item[1], item[2]) for item in ranking]
    pair_position = ranking_pairs.index(fixed_names) + 1
    print(
        f"\nFixed SHAP-LCD pair: {fixed_names[0]} vs {fixed_names[1]} "
        f"(ranking position {pair_position})"
    )

    print("\nAlgorithm 2 - local feature differences Delta S")
    explainer.explain_selected_pairs(
        instance_index,
        selected_pairs=[fixed_names],
        instance_type="test",
        class_names=CLASS_NAMES,
        instance_label=f"Seeds instance {instance_index}",
        save_dir=RESULTS_DIR,
        dpi=600,
        show=SHOW_PLOTS,
    )

    print("\n" + "=" * 72)
    print(" SEEDS LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
