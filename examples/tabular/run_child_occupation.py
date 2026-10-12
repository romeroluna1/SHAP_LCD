"""Local SHAP-LCD example for the four-class child-occupation model.

The script loads the definitive leakage-free artifacts. It does not retrain
the model and does not run robustness metrics; those metrics require the
separate valid-perturbation evaluation in the original feature space.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "child_occupation"
RESULTS_DIR = PROJECT_ROOT / "results" / "child_occupation" / "local"

MODEL_FILE = ARTIFACT_DIR / "random_forest_optimizado.joblib"
X_TEST_FILE = ARTIFACT_DIR / "X_test_final.csv"
Y_TEST_FILE = ARTIFACT_DIR / "y_test_final.csv"
SHAP_FILE = ARTIFACT_DIR / "shap_values_3d.npy"
FEATURE_NAMES_JSON = ARTIFACT_DIR / "feature_names.json"
FEATURE_NAMES_CSV = ARTIFACT_DIR / "feature_names.csv"
METADATA_FILE = ARTIFACT_DIR / "metadata.json"

SEED = 42
INSTANCE_INDEX = 31
TOP_FEATURES = 10
SHOW_PLOTS = True

CLASS_NAMES = {
    0: "asoTrabajo",
    1: "estudia",
    2: "incapacitado",
    3: "otraactividad",
}

# Pairs reported in the child-occupation study.
PAIRS_TO_EXPLAIN = [(0, 1), (0, 3)]


# Keep this exact class name for compatibility with the definitive joblib file.
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
    print(f"ΔP                : {delta_p:.8f}")
    print(f"Sum of feature ΔS : {delta_s.sum():.8f}")
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
    print(" SHAP-LCD - CHILD OCCUPATION - LOCAL TABULAR EXAMPLE")
    print("=" * 76)
    print(f"Artifacts       : {ARTIFACT_DIR}")
    print(f"Results         : {RESULTS_DIR}")
    print(f"Test position   : {INSTANCE_INDEX}")
    print("Fixed pairs     : asoTrabajo vs estudia; asoTrabajo vs otraactividad")
    print("Training        : omitted; definitive validated artifacts are loaded")
    print("Robustness      : omitted; evaluated separately in original space")
    print("=" * 76)

    print("\nLoading definitive artifacts...")
    model = joblib.load(MODEL_FILE)
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
    verify_shap_probability_scale(model, X_test, shap_values)

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

    print("\nAlgorithm 1 - probability differences |ΔP|")
    ranking = probability_ranking(probabilities)
    ranking.to_csv(RESULTS_DIR / "probability_differences.csv", index=False)
    print(ranking.to_string(index=False))
    plot_probability_differences(ranking)

    print("\nAlgorithm 2 - local feature differences ΔS")
    shap_instance = shap_values[INSTANCE_INDEX]
    for class_j, class_r in PAIRS_TO_EXPLAIN:
        explain_pair(
            shap_instance, probabilities, feature_names, class_j, class_r
        )

    if METADATA_FILE.exists():
        print(f"\nExperiment metadata available in: {METADATA_FILE}")

    print("\n" + "=" * 76)
    print(" CHILD-OCCUPATION LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 76)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
