"""SHAP-LCD local de EduFormalDANE (FRUSON, cinco clases).

Carga artefactos definitivos sin reentrenar ni recalcular SHAP.
La evaluacion de estabilidad y consistencia queda en un programa separado.
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
import sys
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "edudane"
RESULTS_DIR = PROJECT_ROOT / "results" / "edudane" / "local"

MODEL_FILE = ARTIFACT_DIR / "random_forest_optimizado.joblib"
X_TEST_FILE = ARTIFACT_DIR / "X_test_final.csv"
Y_TEST_FILE = ARTIFACT_DIR / "y_test_final.csv"
SHAP_FILE = ARTIFACT_DIR / "shap_values_3d.npy"
FEATURE_NAMES_JSON = ARTIFACT_DIR / "feature_names.json"
FEATURE_NAMES_CSV = ARTIFACT_DIR / "feature_names.csv"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"

SEED = 42
INSTANCE_INDEX = 32
TOP_FEATURES = 10
SHOW_PLOTS = True

# Igual que en el main original, los pares se seleccionan atendiendo a |Delta P|.
# Si se requieren pares especificos, usar por ejemplo [(0, 3)].
PAIRS_TO_EXPLAIN = None
TOP_PAIRS_BY_ABS_DELTA_P = 2


def load_class_names() -> dict[int, str]:
    path = ARTIFACT_DIR / "class_names.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        names = {i: str(x) for i, x in enumerate(data)}
    elif isinstance(data, dict):
        names = {int(k): str(v) for k, v in data.items()}
    else:
        raise ValueError("class_names.json debe contener lista o diccionario.")
    if set(names) != set(range(5)):
        raise ValueError(f"Se esperaban cinco clases 0..4: {names}")
    return names


CLASS_NAMES = load_class_names()


# Conservar nombre exacto para compatibilidad con joblib.
class MyOptimizedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, random_state: int = 42):
        super().__init__(
            n_estimators=800,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features=0.5,
            class_weight=None,
            n_jobs=-1,
            random_state=random_state,
        )


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")


def finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Figure saved in: {path}")
    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close(fig)


def fix_mojibake(text: str) -> str:
    """Corrige caracteres danados por una interpretaciÃ³n incorrecta de UTF-8."""
    if not isinstance(text, str):
        return text

    text = text.replace("\ufeff", "").replace("Ã¯Â»Â¿", "")

    for _ in range(2):
        try:
            corrected = text.encode("latin1").decode("utf-8")
            text = corrected
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

    return text

def load_feature_names(n_features: int) -> list[str]:
    names: list[str] = []

    if FEATURE_NAMES_JSON.exists():
        with FEATURE_NAMES_JSON.open("r", encoding="utf-8") as file:
            obj = json.load(file)

        if isinstance(obj, list):
            names = [str(value) for value in obj]
        elif isinstance(obj, dict):
            for key in ("feature_names", "features", "columns"):
                if key in obj and isinstance(obj[key], list):
                    names = [str(value) for value in obj[key]]
                    break
            if not names and all(str(key).isdigit() for key in obj):
                names = [str(obj[key]) for key in sorted(obj, key=lambda x: int(x))]

    if not names and FEATURE_NAMES_CSV.exists():
        table = pd.read_csv(FEATURE_NAMES_CSV)
        if "feature_name" in table.columns:
            names = table["feature_name"].astype(str).tolist()
        elif table.shape[1] == 1:
            names = table.iloc[:, 0].astype(str).tolist()

    if len(names) != n_features:
        raise ValueError(
            "Feature-name count does not match X_test: "
            f"names={len(names)}, features={n_features}."
        )
    names = [fix_mojibake(name) for name in names]
    return names


def normalize_shap_shape(
    values: np.ndarray,
    n_instances: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    arr = np.asarray(values)
    expected = (n_instances, n_features, n_classes)
    if arr.shape == expected:
        return arr
    if arr.shape == (n_classes, n_instances, n_features):
        return np.transpose(arr, (1, 2, 0))
    if arr.shape == (n_instances, n_classes, n_features):
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Unexpected SHAP shape {arr.shape}; expected {expected}.")


def verify_shap_probability_scale(
    model,
    X_test: np.ndarray,
    shap_values: np.ndarray,
    n_checks: int = 10,
    tolerance: float = 1e-6,
) -> None:
    tree_explainer = shap.TreeExplainer(model)
    base = np.asarray(tree_explainer.expected_value, dtype=float).reshape(-1)
    checks = min(n_checks, len(X_test))
    reconstructed = base[None, :] + shap_values[:checks].sum(axis=1)
    predicted = model.predict_proba(X_test[:checks])
    errors = np.abs(reconstructed - predicted)
    max_error = float(errors.max())

    print("\n" + "=" * 72)
    print(" SHAP PROBABILITY-SCALE VERIFICATION")
    print("=" * 72)
    print("Explained output : predict_proba")
    print(f"Checked instances: {checks}")
    print(f"Mean error       : {errors.mean():.3e}")
    print(f"Maximum error    : {max_error:.3e}")
    if max_error > tolerance:
        raise RuntimeError(
            "Cached SHAP values do not reconstruct the loaded model's "
            "probabilities. Execution stopped to avoid mixing artifacts."
        )
    print("VERIFICATION PASSED: SHAP values are on the probability scale.")
    print("=" * 72)


def probability_ranking(probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for class_j, class_r in combinations(sorted(CLASS_NAMES), 2):
        delta = float(probabilities[class_j] - probabilities[class_r])
        rows.append(
            {
                "class_j_index": class_j,
                "class_j": CLASS_NAMES[class_j],
                "class_r_index": class_r,
                "class_r": CLASS_NAMES[class_r],
                "probability_j": float(probabilities[class_j]),
                "probability_r": float(probabilities[class_r]),
                "delta_probability": delta,
                "abs_delta_probability": abs(delta),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_delta_probability", ascending=False
    ).reset_index(drop=True)


def plot_probability_differences(ranking: pd.DataFrame) -> None:
    plot_df = ranking.copy()

    def wrap_if_long(name: str, max_len: int = 20) -> str:
        name = str(name)
        if len(name) <= max_len:
            return name
        middle = len(name) // 2
        left = name.rfind(" ", 0, middle)
        right = name.find(" ", middle)
        if left == -1 and right == -1:
            return name[:max_len] + "\n" + name[max_len:]
        cut = left if left != -1 and (right == -1 or middle - left <= right - middle) else right
        return name[:cut].strip() + "\n" + name[cut:].strip()

    class_labels = [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)]
    use_multiline = any(len(label) > 10 for label in class_labels)
    if use_multiline:
        labels = [
            f"{wrap_if_long(row.class_j)}\nvs\n{wrap_if_long(row.class_r)}"
            for row in plot_df.itertuples()
        ]
    else:
        labels = [
            f"{row.class_j} vs {row.class_r}"
            for row in plot_df.itertuples()
        ]

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(
        x=plot_df["abs_delta_probability"],
        y=labels,
        hue=labels,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for tick in ax.get_yticklabels():
        tick.set_multialignment("center")
        tick.set_linespacing(0.85)
        tick.set_ha("right")
    ax.set_title(
        f"Absolute Differences Between Class Probabilities\n"
        f"for Test Instance {INSTANCE_INDEX}",
        fontsize=34,
    )
    ax.set_xlabel(r"Absolute Differences $|\mathrm{\Delta P}|$", fontsize=34)
    ax.set_ylabel("Class Pairs", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    finish_figure(fig, RESULTS_DIR / "probability_differences.png")


def explain_pair(
    shap_instance: np.ndarray,
    probabilities: np.ndarray,
    feature_names: list[str],
    class_j: int,
    class_r: int,
) -> None:
    delta_s = shap_instance[:, class_j] - shap_instance[:, class_r]
    delta_p = float(probabilities[class_j] - probabilities[class_r])
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_class_j": shap_instance[:, class_j],
            "shap_class_r": shap_instance[:, class_r],
            "delta_shap": delta_s,
            "abs_delta_shap": np.abs(delta_s),
        }
    ).sort_values("abs_delta_shap", ascending=False)

    pair_tag = (
        f"{safe_filename(CLASS_NAMES[class_j])}_vs_"
        f"{safe_filename(CLASS_NAMES[class_r])}"
    )
    table.to_csv(RESULTS_DIR / f"deltaS_{pair_tag}.csv", index=False)
    top = table.head(TOP_FEATURES)

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.barplot(
        x="delta_shap",
        y="feature",
        hue="feature",
        data=top,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        f"Differences in SHAP Values of Main Features\n"
        f"Between '{CLASS_NAMES[class_j]}' and '{CLASS_NAMES[class_r]}' "
        f"for Test Instance {INSTANCE_INDEX}",
        fontsize=30,
    )
    ax.set_xlabel(r"SHAP Differences $\mathrm{\Delta S}$", fontsize=30)
    ax.set_ylabel("Features in Descending Order", fontsize=30)
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)
    finish_figure(fig, RESULTS_DIR / f"deltaS_{pair_tag}.png")

    print("\nLocal SHAP-LCD explanation")
    print(f"Pair              : {CLASS_NAMES[class_j]} vs {CLASS_NAMES[class_r]}")
    print(f"Delta P                : {delta_p:.8f}")
    print(f"Sum of feature Delta S : {delta_s.sum():.8f}")
    print("The class-specific base-value difference completes the reconstruction.")


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [MODEL_FILE, X_TEST_FILE, Y_TEST_FILE, SHAP_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n - " + "\n - ".join(missing))

    print("Python          :", sys.version.splitlines()[0])
    print("NumPy           :", np.__version__)
    print("pandas          :", pd.__version__)
    print("scikit-learn    :", sklearn.__version__)
    print("SHAP            :", shap.__version__)
    print("matplotlib      :", matplotlib.__version__)
    print("OS              :", platform.platform())

    print("\n" + "=" * 76)
    print(" SHAP-LCD - EDUFORMALDANE - LOCAL TABULAR EXAMPLE")
    print("=" * 76)
    print(f"Artifacts       : {ARTIFACT_DIR}")
    print(f"Results         : {RESULTS_DIR}")
    print(f"Test position   : {INSTANCE_INDEX}")
    print("Pair selection  : interactive by |Delta P|" if PAIRS_TO_EXPLAIN is None else f"Pair selection  : {PAIRS_TO_EXPLAIN}")
    print("Training        : omitted; definitive validated artifacts are loaded")
    print("Robustness      : omitted; evaluated separately in original space")
    print("=" * 76)

    print("\nLoading definitive artifacts...")
    model = joblib.load(MODEL_FILE)
    if list(map(int, model.classes_)) != sorted(CLASS_NAMES):
        raise ValueError(f"Clases del modelo {model.classes_} no coinciden con {CLASS_NAMES}")
    X_test = pd.read_csv(X_TEST_FILE).to_numpy(dtype=float)
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, -1].to_numpy(dtype=int)
    raw_shap = np.load(SHAP_FILE, allow_pickle=False)

    n_instances, n_features = X_test.shape
    feature_names = load_feature_names(n_features)
    shap_values = normalize_shap_shape(
        raw_shap, n_instances, n_features, len(CLASS_NAMES)
    )
    if len(y_test) != n_instances:
        raise ValueError("X_test and y_test have different row counts.")
    if not 0 <= INSTANCE_INDEX < n_instances:
        raise IndexError(f"INSTANCE_INDEX must be between 0 and {n_instances - 1}.")

    print(f"X_test shape    : {X_test.shape}")
    print(f"SHAP shape      : {shap_values.shape}")
    print(f"Feature count   : {len(feature_names)}")
    if not np.isfinite(X_test).all() or not np.isfinite(shap_values).all():
        raise ValueError("X_test o los valores SHAP contienen valores no finitos.")
    verify_shap_probability_scale(model, X_test, shap_values)

    # ============================================================
    # EVALUACION DEL MODELO DEFINITIVO EN TEST (SIN CV NI REFIT)
    # ============================================================
    print("\n" + "=" * 76)
    print(" MODEL EVALUATION - DEFINITIVE TEST SET")
    print("=" * 76)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro-Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Macro-Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Macro-F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    print(f"Test instances : {len(y_test)}")
    for metric, value in metrics.items():
        print(f"{metric:<18}: {value:.6f}")

    metrics_df = pd.DataFrame([{
        "dataset": "EduFormalDANE",
        "evaluation": "Final held-out test set",
        "n_test": len(y_test),
        **metrics,
    }])
    metrics_path = RESULTS_DIR / "model_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved in: {metrics_path}")
    print("=" * 76)

    x_instance = X_test[INSTANCE_INDEX : INSTANCE_INDEX + 1]
    probabilities_raw = model.predict_proba(x_instance)[0]
    probabilities = np.zeros(len(CLASS_NAMES), dtype=float)
    for class_id, probability in zip(model.classes_, probabilities_raw):
        probabilities[int(class_id)] = float(probability)

    true_class = int(y_test[INSTANCE_INDEX])
    predicted_class = int(np.argmax(probabilities))
    print("\nSelected local instance")
    print(f"Test position   : {INSTANCE_INDEX}")
    print(f"True class      : {CLASS_NAMES[true_class]}")
    print(f"Predicted class : {CLASS_NAMES[predicted_class]}")
    for class_id in sorted(CLASS_NAMES):
        print(f"P({CLASS_NAMES[class_id]}) = {probabilities[class_id]:.8f}")

    pd.DataFrame([{
        "test_position": INSTANCE_INDEX,
        "true_class_index": true_class,
        "true_class": CLASS_NAMES[true_class],
        "predicted_class_index": predicted_class,
        "predicted_class": CLASS_NAMES[predicted_class],
        **{f"probability_{CLASS_NAMES[c]}": probabilities[c] for c in CLASS_NAMES},
    }]).to_csv(RESULTS_DIR / "selected_instance.csv", index=False)

    print("\nAlgorithm 1 - probability differences |Delta P|")
    ranking = probability_ranking(probabilities)
    ranking.to_csv(RESULTS_DIR / "probability_differences.csv", index=False)
    print(ranking.to_string(index=False))
    plot_probability_differences(ranking)

    print("\nAlgorithm 2 - local feature differences Delta S")
    shap_instance = shap_values[INSTANCE_INDEX]

    pairs = PAIRS_TO_EXPLAIN

    if pairs is None:
        print("\nPARES DE CLASES ORDENADOS POR |Delta P|:")
        available_pairs = []

        for number, row in enumerate(ranking.itertuples(), start=1):
            class_j = int(row.class_j_index)
            class_r = int(row.class_r_index)
            available_pairs.append((class_j, class_r))
            print(
                f"{number}. {CLASS_NAMES[class_j]} vs {CLASS_NAMES[class_r]} "
                f"|Delta P| = {row.abs_delta_probability:.6f}"
            )

        while True:
            selection = input(
                "\nIntroduce los numeros de los pares que deseas explicar "
                "(ejemplo: 1,3): "
            ).strip()

            try:
                selected_numbers = [
                    int(value.strip())
                    for value in selection.split(",")
                ]
                if (
                    not selected_numbers
                    or len(set(selected_numbers)) != len(selected_numbers)
                    or any(
                        number < 1 or number > len(available_pairs)
                        for number in selected_numbers
                    )
                ):
                    raise ValueError

                pairs = [
                    available_pairs[number - 1]
                    for number in selected_numbers
                ]
                break
            except ValueError:
                print(
                    "Seleccion no valida. Introduce numeros distintos "
                    "de la lista, separados por comas."
                )
    for class_j, class_r in pairs:
        if class_j not in CLASS_NAMES or class_r not in CLASS_NAMES or class_j == class_r:
            raise ValueError(f"Par de clases no valido: {(class_j, class_r)}")
        explain_pair(
            shap_instance, probabilities, feature_names, class_j, class_r
        )

    if METADATA_FILE.exists():
        print(f"\nExperiment metadata available in: {METADATA_FILE}")

    print("\n" + "=" * 76)
    print(" EDUFORMALDANE LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 76)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

