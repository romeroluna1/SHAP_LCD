"""Reproducible SHAP-LCD experiment for Fashion-MNIST."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =====================================================================
# 1) Repository paths and package import
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd.image import SHAPExplainer2
from shap_lcd.image_metrics import evaluate_pair_metrics_image


# =====================================================================
# 2) Experiment configuration
# =====================================================================
SEED = 42
QUICK_TEST = False

# Paper instance in the official Fashion-MNIST test set.
PAPER_INSTANCE_INDEX = 6838
PAPER_LABEL = "Instance 6838"

# Only the paper instance is needed for the local figures. A reproducible
# subset of 10 images (always containing instance 6838) avoids calculating
# unused SHAP values for all 10,000 test images. Robustness SHAP values are
# calculated independently below for every evaluated image and perturbation.
SHAP_EVAL_SIZE = 10

# Robustness protocol. The article reports both configurations for both pairs.
N_METRICS = 2 if QUICK_TEST else 50
M_PERTURBATIONS = 2 if QUICK_TEST else 40
TOP_K = None
ROBUSTNESS_PAIRS = [
    ("T-shirt/Top", "Trouser"),
    ("Trouser", "Coat"),
]
ROBUSTNESS_CONFIGURATIONS = [
    {"name": "configuration_1", "sigma": 0.01, "tau": 0.85},
    {"name": "configuration_2", "sigma": 0.02, "tau": 0.90},
]

# Select a pair interactively after ranking by |Delta P|.
# Press Enter to use the first pair in the ranking.
PAIR_RANK_DEFAULT = 1

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "fashion_mnist"
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "fashion_mnist"
    / ("quick_test" if QUICK_TEST else "final")
)
MODEL_PATH = ARTIFACT_DIR / "cnn_fashion.keras"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 3) Reproducibility
# =====================================================================
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("Deterministic operations enabled.")
except Exception as error:
    print("Deterministic operations are unavailable:", error)


CLASS_NAMES = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, data_generator, x_train, y_train, x_validation, y_validation):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        min_delta=1e-4,
        restore_best_weights=True,
        mode="max",
    )
    return model.fit(
        data_generator.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_validation, y_validation),
        callbacks=[early_stopping],
        verbose=1,
    )


def save_training_curves(history):
    specifications = [
        ("loss", "val_loss", "Loss", "loss_curve.png"),
        ("accuracy", "val_accuracy", "Accuracy", "accuracy_curve.png"),
    ]
    for train_key, validation_key, ylabel, filename in specifications:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history.history[train_key], label=f"Train {ylabel.lower()}")
        axis.plot(history.history[validation_key], label=f"Validation {ylabel.lower()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} curve")
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.4)
        figure.tight_layout()
        figure.savefig(RESULT_DIR / filename, dpi=600, bbox_inches="tight")
        plt.close(figure)


def print_model_metrics(model, x_test, y_test):
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print("\nFinal model metrics on the official test set")
    print(f"Accuracy           : {accuracy_score(y_test, predictions):.6f}")
    print(f"Balanced accuracy  : {balanced_accuracy_score(y_test, predictions):.6f}")
    print(f"F1-score macro     : {f1_score(y_test, predictions, average='macro'):.6f}")
    print(f"MCC                : {matthews_corrcoef(y_test, predictions):.6f}")
    print(
        f"Macro-AUC (OvO)    : "
        f"{roc_auc_score(y_test, probabilities, multi_class='ovo', average='macro'):.6f}"
    )


def prepare_evaluation_subset(x_test, y_test, size, random_generator):
    if size is None:
        return x_test, y_test, PAPER_INSTANCE_INDEX, None, "ALL"
    if size < 2 or size > len(x_test):
        raise ValueError(f"SHAP_EVAL_SIZE debe estar entre 2 y {len(x_test)}.")

    all_indices = np.arange(len(x_test))
    remaining = np.setdiff1d(all_indices, np.asarray([PAPER_INSTANCE_INDEX]))
    additional = random_generator.choice(remaining, size=size - 1, replace=False)
    selected = np.concatenate([[PAPER_INSTANCE_INDEX], additional])
    return x_test[selected], y_test[selected], 0, size, f"M{size}"


def select_pair(probability_differences):
    for position, (difference, class_i, class_j) in enumerate(
        probability_differences[:10], start=1
    ):
        print(f"{position}. {class_i} vs {class_j}: |Delta P| = {difference:.6f}")

    answer = input(
        f"\nPair number [Enter = {PAIR_RANK_DEFAULT}]: "
    ).strip()
    selected_rank = PAIR_RANK_DEFAULT if answer == "" else int(answer)
    if not 1 <= selected_rank <= len(probability_differences):
        raise ValueError("El número del par seleccionado no es válido.")
    selected = probability_differences[selected_rank - 1]
    return selected[1], selected[2]


def main():
    print("Python executable :", sys.executable)
    print("TensorFlow        :", tf.__version__)
    print("Keras             :", keras.__version__)
    print("Project root      :", PROJECT_ROOT)
    print("Quick test        :", QUICK_TEST)
    print("Results           :", RESULT_DIR)

    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train_full = x_train_full[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.10,
        stratify=y_train_full,
        random_state=SEED,
    )

    data_generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    data_generator.fit(x_train)

    if MODEL_PATH.exists():
        print("\nLoading model:", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("\nModel not found; training a new CNN.")
        model = build_model()
        history = train_model(
            model,
            data_generator,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        model.save(MODEL_PATH)
        save_training_curves(history)
        print("Model saved in:", MODEL_PATH)

    print_model_metrics(model, x_test, y_test)

    random_generator = np.random.default_rng(SEED)
    background_indices = random_generator.choice(
        len(x_train), size=50, replace=False
    )
    background = x_train[background_indices].astype(np.float32)

    x_evaluation, y_evaluation, instance_index, desired_size, pickle_tag = (
        prepare_evaluation_subset(
            x_test, y_test, SHAP_EVAL_SIZE, random_generator
        )
    )
    pickle_path = ARTIFACT_DIR / f"shap_values_fashion_paper_{pickle_tag}.pickle"

    explainer = SHAPExplainer2(
        model=model,
        x_background=background,
        x_eval=x_evaluation,
        y_eval=y_evaluation,
        framework="keras",
        random_state=SEED,
    )

    if pickle_path.exists():
        explainer.load_shap_values(pickle_path)
        compatible, reason = explainer.is_cache_compatible(desired_size)
        if compatible:
            print("Compatible SHAP cache loaded:", pickle_path)
        else:
            print(f"Incompatible SHAP cache ({reason}); recalculating.")
            explainer.calculate_shap_values(max_eval=desired_size)
            explainer.save_shap_values(pickle_path)
    else:
        explainer.calculate_shap_values(max_eval=desired_size)
        explainer.save_shap_values(pickle_path)
        print("SHAP values saved in:", pickle_path)

    probability_differences = explainer.plot_probability_differences(
        instance_idx=instance_index,
        class_names=CLASS_NAMES,
        instance_label=PAPER_LABEL,
        top_k=10,
        save_path=RESULT_DIR / "probability_differences_instance_6838.png",
        show=False,
    )
    class_i_name, class_j_name = select_pair(probability_differences)
    selected_pairs = [(class_i_name, class_j_name)]

    explainer.explain_selected_pairs(
        instance_idx=instance_index,
        selected_pairs=selected_pairs,
        class_names=CLASS_NAMES,
        instance_label=PAPER_LABEL,
        top_n=7,
        output_dir=RESULT_DIR,
        show=False,
    )
    safe_i = class_i_name.replace("/", "_").replace(" ", "_")
    safe_j = class_j_name.replace("/", "_").replace(" ", "_")
    explainer.plot_shap_heatmap(
        instance_idx=instance_index,
        class_i=class_i_name,
        class_j=class_j_name,
        class_names=CLASS_NAMES,
        instance_label=PAPER_LABEL,
        cmap="PiYG",
        alpha=0.6,
        show_original=True,
        save_path=RESULT_DIR / f"heatmap_{safe_i}_vs_{safe_j}.png",
        show=False,
    )

    metric_columns = [
        "Stability_S",
        "Consistency_C_predexp",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]
    all_summaries = []

    for configuration in ROBUSTNESS_CONFIGURATIONS:
        configuration_dir = RESULT_DIR / configuration["name"]
        configuration_dir.mkdir(parents=True, exist_ok=True)

        for metric_class_i, metric_class_j in ROBUSTNESS_PAIRS:
            print("\n" + "=" * 72)
            print(
                f"Robustness: {metric_class_i} vs {metric_class_j} | "
                f"sigma={configuration['sigma']} | tau={configuration['tau']}"
            )
            print("=" * 72)

            instance_results, summary = evaluate_pair_metrics_image(
                model=model,
                shap_explainer=explainer.explainer,
                X_test=x_test,
                y_test=y_test,
                class_names=CLASS_NAMES,
                class_i_name=metric_class_i,
                class_j_name=metric_class_j,
                filter_mode="true",
                N_max=N_METRICS,
                M_pert=M_PERTURBATIONS,
                sigma=configuration["sigma"],
                tau=configuration["tau"],
                top_k=TOP_K,
                seed=SEED,
            )

            summary = summary.T
            summary["median"] = instance_results[metric_columns].median()
            summary["SE"] = summary["std"] / np.sqrt(len(instance_results))
            summary["mean+-std"] = summary.apply(
                lambda row: f"{row['mean']:.6f} +- {row['std']:.6f}", axis=1
            )

            pair_tag = (
                f"{metric_class_i}_vs_{metric_class_j}"
                .replace("/", "_")
                .replace(" ", "_")
            )
            instance_results.to_csv(
                configuration_dir / f"metrics_instances_{pair_tag}.csv",
                index=False,
            )
            summary.to_csv(
                configuration_dir / f"metrics_summary_{pair_tag}.csv"
            )

            for metric_name, row in summary.iterrows():
                all_summaries.append(
                    {
                        "configuration": configuration["name"],
                        "sigma": configuration["sigma"],
                        "tau": configuration["tau"],
                        "pair": f"{metric_class_i} vs {metric_class_j}",
                        "N": len(instance_results),
                        "M": M_PERTURBATIONS,
                        "metric": metric_name,
                        "mean": row["mean"],
                        "std": row["std"],
                        "median": row["median"],
                        "SE": row["SE"],
                    }
                )

            print("\nResults for this pair")
            print(summary.to_string())

    combined_summary = pd.DataFrame(all_summaries)
    combined_summary.to_csv(
        RESULT_DIR / "metrics_summary_all_configurations.csv", index=False
    )

    print("\nCombined robustness results")
    print(combined_summary.to_string(index=False))
    print("\nExecution completed.")
    print("Results saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()
