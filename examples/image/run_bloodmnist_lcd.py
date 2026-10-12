"""Reproducible local SHAP-LCD example for BloodMNIST."""

from __future__ import annotations

import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from medmnist import BloodMNIST


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd.image import SHAPExplainer2  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------
SEED = 42
CLASS_J = 6  # Neutrophil
CLASS_R = 1  # Eosinophil
BACKGROUND_SIZE = 100
TOP_PAIRS = 10
TOP_PIXELS = 10
SHOW_PLOTS = True

CLASS_NAMES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet",
}

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "bloodmnist"
MODEL_PATH = ARTIFACT_DIR / "bloodmnist_resnet18_best.keras"
SHAP_PATH = (
    ARTIFACT_DIR
    / "shap_values_bloodmnist_resnet18_best_6_vs_1.pickle"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "bloodmnist" / "local"


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        print("Deterministic operations enabled.")
    except Exception as exc:
        print(f"Deterministic operations unavailable: {exc}")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = BloodMNIST(split=split, download=True)
    images = dataset.imgs.astype("float32") / 255.0
    labels = np.asarray(dataset.labels).reshape(-1).astype("int64")
    return images, labels


def load_cache_metadata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError("The BloodMNIST SHAP pickle must be a dictionary.")
    if "shap_values" not in payload or "idx_eval" not in payload:
        raise ValueError("The SHAP pickle must contain shap_values and idx_eval.")
    return payload["shap_values"], np.asarray(payload["idx_eval"], dtype=int)


def choose_local_instance(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
) -> int:
    """Choose a correctly classified pair image informative for both classes."""
    predictions = np.argmax(probabilities, axis=1)
    belongs_to_pair = (true_labels == CLASS_J) | (true_labels == CLASS_R)
    correctly_classified = predictions == true_labels
    candidates = np.flatnonzero(belongs_to_pair & correctly_classified)
    if candidates.size == 0:
        raise ValueError("No correctly classified cached image exists for the pair.")

    # A large minimum gives an image where both selected-class probabilities
    # participate meaningfully in the comparison.
    pair_scores = np.minimum(
        probabilities[candidates, CLASS_J],
        probabilities[candidates, CLASS_R],
    )
    return int(candidates[np.argmax(pair_scores)])


def save_original_image(
    image: np.ndarray,
    title: str,
    path: Path,
    show: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image)
    axis.set_title(title, fontsize=13)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"Original image saved in: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [path for path in (MODEL_PATH, SHAP_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing BloodMNIST artifacts:\n - "
            + "\n - ".join(str(path) for path in missing)
        )

    print("=" * 76)
    print(" SHAP-LCD - BLOODMNIST - LOCAL IMAGE EXAMPLE")
    print("=" * 76)
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Model           : {MODEL_PATH}")
    print(f"SHAP pickle     : {SHAP_PATH}")
    print(
        f"Fixed pair      : {CLASS_NAMES[CLASS_J]} vs "
        f"{CLASS_NAMES[CLASS_R]}"
    )
    print(f"Results         : {RESULTS_DIR}")
    print(f"Show plots      : {SHOW_PLOTS}")
    print("=" * 76)

    print("\nLoading BloodMNIST...")
    x_train, _ = load_split("train")
    x_test, y_test = load_split("test")
    print(f"Train: {x_train.shape}")
    print(f"Test : {x_test.shape} | {y_test.shape}")

    print(f"\nLoading definitive model:\n{MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nLoading cached SHAP indices...")
    shap_values, idx_eval = load_cache_metadata(SHAP_PATH)
    x_eval = x_test[idx_eval]
    y_eval = y_test[idx_eval]
    print(f"Cached images: {x_eval.shape}")
    print(f"Cached SHAP shape: {np.asarray(shap_values).shape}")

    print("\nCalculating predictions for the cached pair sample...")
    probabilities = model.predict(x_eval, batch_size=64, verbose=1)
    local_position = choose_local_instance(probabilities, y_eval)
    test_index = int(idx_eval[local_position])
    true_label = int(y_eval[local_position])
    predicted_label = int(np.argmax(probabilities[local_position]))

    print("\nSelected local image")
    print(f"Cache position  : {local_position}")
    print(f"Test index      : {test_index}")
    print(f"True class      : {CLASS_NAMES[true_label]}")
    print(f"Predicted class : {CLASS_NAMES[predicted_label]}")
    print(
        f"P({CLASS_NAMES[CLASS_J]}) = "
        f"{probabilities[local_position, CLASS_J]:.6f}"
    )
    print(
        f"P({CLASS_NAMES[CLASS_R]}) = "
        f"{probabilities[local_position, CLASS_R]:.6f}"
    )

    metadata = pd.DataFrame(
        [
            {
                "cache_position": local_position,
                "test_index": test_index,
                "true_class_id": true_label,
                "true_class": CLASS_NAMES[true_label],
                "predicted_class_id": predicted_label,
                "predicted_class": CLASS_NAMES[predicted_label],
                "probability_neutrophil": probabilities[
                    local_position, CLASS_J
                ],
                "probability_eosinophil": probabilities[
                    local_position, CLASS_R
                ],
            }
        ]
    )
    metadata.to_csv(
        RESULTS_DIR / "selected_image_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    probability_table = pd.DataFrame(
        {
            "class_id": sorted(CLASS_NAMES),
            "class_name": [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)],
            "probability": probabilities[local_position],
        }
    )
    probability_table.to_csv(
        RESULTS_DIR / "selected_image_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_original_image(
        x_eval[local_position],
        (
            f"BloodMNIST test image {test_index}\n"
            f"True: {CLASS_NAMES[true_label]} | "
            f"Predicted: {CLASS_NAMES[predicted_label]}"
        ),
        RESULTS_DIR / "selected_original_image.png",
        SHOW_PLOTS,
    )

    # Reconstruct the same background rule used to create the global cache.
    np.random.seed(SEED)
    background_indices = np.random.choice(
        len(x_train),
        size=BACKGROUND_SIZE,
        replace=False,
    )
    background = x_train[background_indices]

    # GradientExplainer matches the explainer used for the existing cache.
    explainer = SHAPExplainer2(
        model=model,
        x_background=background,
        x_eval=x_eval,
        y_eval=y_eval,
        framework="keras",
        explainer_type="gradient",
        random_state=SEED,
    )
    explainer.load_shap_values(SHAP_PATH)

    instance_label = f"BloodMNIST test image {test_index}"

    print("\nAlgorithm 1 - probability differences |Delta P|")
    ranking = explainer.plot_probability_differences(
        instance_idx=local_position,
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        top_k=TOP_PAIRS,
        save_path=RESULTS_DIR / "probability_differences.png",
        show=SHOW_PLOTS,
    )
    for position, (difference, class_a, class_b) in enumerate(ranking, start=1):
        print(
            f"{position:2d}. {class_a} vs {class_b}: "
            f"|Delta P|={difference:.6f}"
        )

    fixed_pair = (CLASS_NAMES[CLASS_J], CLASS_NAMES[CLASS_R])
    fixed_pair_rank = next(
        position
        for position, (_, class_a, class_b) in enumerate(ranking, start=1)
        if {class_a, class_b} == set(fixed_pair)
    )
    print(
        f"\nFixed SHAP-LCD pair: {fixed_pair[0]} vs {fixed_pair[1]} "
        f"(ranking position {fixed_pair_rank})"
    )

    print("\nAlgorithm 2 - local pixel differences Delta S")
    explainer.explain_selected_pairs(
        instance_idx=local_position,
        selected_pairs=[fixed_pair],
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        top_n=TOP_PIXELS,
        output_dir=RESULTS_DIR,
        show=SHOW_PLOTS,
    )

    print("\nAlgorithm 3 - SHAP-LCD heatmap")
    explainer.plot_shap_heatmap(
        instance_idx=local_position,
        class_i=fixed_pair[0],
        class_j=fixed_pair[1],
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        cmap="PiYG",
        alpha=0.60,
        show_original=True,
        save_path=RESULTS_DIR / "heatmap_Neutrophil_vs_Eosinophil.png",
        show=SHOW_PLOTS,
    )

    print("\n" + "=" * 76)
    print(" BLOODMNIST LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 76)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
