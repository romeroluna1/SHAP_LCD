"""Local SHAP-LCD experiment for HAM10000 (monograph extension).

This experiment reuses the lesion-level split, ResNet50 weights and SHAP
values produced by the definitive SHAP-GLCD HAM10000 experiment. It does not
retrain the classifier and does not recompute the cached local SHAP values.
"""

from __future__ import annotations

import gc
import os
import pickle
import random
import re
import sys
from pathlib import Path

import kagglehub
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
from tensorflow.keras import layers, models


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

# True: small reproducibility test. False: definitive experiment.
QUICK_TEST = True

# False generates only the local SHAP-LCD figures. True additionally runs the
# expensive stability/consistency evaluation.
RUN_ROBUSTNESS = False

IMG_SIZE = 224
BATCH_SIZE = 16
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Local SHAP-LCD pair used in the monograph extension.
CLASS_J = 1  # BCC
CLASS_R = 2  # BKL
LOCAL_PAIR = ("BCC", "BKL")

# The local case is selected deterministically from the cached sample: the
# first correctly classified BCC image in ascending original test-index order.
LOCAL_TRUE_CLASS = CLASS_J

# Reproducible training-only background for GradientExplainer.
BACKGROUND_SIZE = 30

# Robustness protocol.
N_METRICS_PER_CLASS = 1 if QUICK_TEST else 25
M_PERTURBATIONS = 2 if QUICK_TEST else 40
TOP_K = None
ROBUSTNESS_CONFIGURATIONS = [
    {"name": "configuration_1", "sigma": 0.01, "tau": 0.85},
    {"name": "configuration_2", "sigma": 0.02, "tau": 0.90},
]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "ham10000"
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ham10000"
    / ("quick_test" if QUICK_TEST else "final")
)

WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "ham10000_resnet50_best_overall_lesion_split_030926.weights.h5"
)
SHAP_PICKLE_PATH = (
    ARTIFACT_DIR
    / "shap_values_ham10000_resnet50_lesion_split_030926_1_vs_2.pickle"
)
MANIFEST_PATH = (
    ARTIFACT_DIR
    / "ham10000_split_manifest_by_lesion_030926.csv"
)

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


CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Short names are used in local figures.
CLASS_NAMES = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}


# =====================================================================
# 4) Data and model helpers
# =====================================================================
def safe_name(text):
    text = str(text).replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def build_image_path_map(root):
    path_map = {}
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        for image_path in Path(root).rglob(extension):
            path_map[image_path.stem] = str(image_path)
    return path_map


def decode_resize_image(path):
    image = tf.io.read_file(str(path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return (tf.cast(image, tf.float32) / 255.0).numpy()


def load_images(dataframe):
    images = []
    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        if position == 1 or position % 50 == 0 or position == len(dataframe):
            print(f"Loading image {position}/{len(dataframe)}")
        images.append(decode_resize_image(row.path))
    return np.stack(images).astype(np.float32)


def make_dataset(dataframe):
    paths = dataframe["path"].astype(str).to_numpy()
    labels = dataframe["label"].astype(np.int64).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return (
        dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_resnet50_ham10000():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.03, 0.03),
            layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )
    x = augmentation(inputs)
    x = layers.Rescaling(255.0, name="to_255")(x)
    x = layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess",
    )(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(7, activation="softmax", name="classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_HAM10000")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_model_metrics(model, test_dataframe):
    dataset = make_dataset(test_dataframe)
    probabilities = model.predict(dataset, verbose=1)
    labels = test_dataframe["label"].to_numpy(dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)

    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovo",
            average="macro",
            labels=np.arange(7),
        )
    except ValueError:
        auc = np.nan

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "mcc": matthews_corrcoef(labels, predictions),
        "macro_auc_ovo": auc,
    }
    print("\nFinal model metrics on the independent test set")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.6f}")
    pd.DataFrame([metrics]).to_csv(RESULT_DIR / "model_test_metrics.csv", index=False)
    return probabilities, predictions


def load_manifest_with_current_paths():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    print("\nLocating HAM10000 with KaggleHub...")
    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    path_map = build_image_path_map(dataset_root)
    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"image_id", "lesion_id", "dx", "label", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")

    manifest["path"] = manifest["image_id"].map(path_map)
    if manifest["path"].isna().any():
        count = int(manifest["path"].isna().sum())
        raise FileNotFoundError(f"Current path not found for {count} images.")

    train_df = manifest[manifest["split"] == "train"].copy().reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    return train_df, test_df


def balanced_metric_sample(test_dataframe):
    parts = []
    for class_id in (CLASS_J, CLASS_R):
        available = test_dataframe[test_dataframe["label"] == class_id]
        if len(available) < N_METRICS_PER_CLASS:
            raise ValueError(
                f"Only {len(available)} images are available for class {class_id}; "
                f"{N_METRICS_PER_CLASS} were requested."
            )
        parts.append(
            available.sample(n=N_METRICS_PER_CLASS, random_state=SEED)
        )
    selected = pd.concat(parts, ignore_index=True)
    return selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def select_local_position(cached_dataframe, predictions):
    candidates = np.where(
        (cached_dataframe["label"].to_numpy() == LOCAL_TRUE_CLASS)
        & (predictions == LOCAL_TRUE_CLASS)
    )[0]
    if len(candidates) == 0:
        candidates = np.where(
            cached_dataframe["label"].to_numpy() == LOCAL_TRUE_CLASS
        )[0]
    if len(candidates) == 0:
        raise ValueError("No cached local image exists for the requested class.")

    original_indices = cached_dataframe.iloc[candidates]["test_index"].to_numpy()
    return int(candidates[np.argmin(original_indices)])


def verify_output(path, description):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{description} was not created correctly: {path}")
    print(f"{description} saved in: {path}")


# =====================================================================
# 5) Main experiment
# =====================================================================
def main():
    print("=" * 76)
    print("SHAP-LCD - HAM10000 - LOCAL IMAGE EXPERIMENT")
    print("=" * 76)
    print("Python executable :", sys.executable)
    print("TensorFlow        :", tf.__version__)
    print("Project root      :", PROJECT_ROOT)
    print("Quick test        :", QUICK_TEST)
    print("Run robustness    :", RUN_ROBUSTNESS)
    print("Local pair        : BCC vs BKL")
    print("Results           :", RESULT_DIR)
    print("=" * 76)

    for required_path in (WEIGHTS_PATH, SHAP_PICKLE_PATH, MANIFEST_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Required artifact not found: {required_path}")

    train_df, test_df = load_manifest_with_current_paths()
    print("\nImages by split")
    print("Train:", len(train_df))
    print("Test :", len(test_df))
    print("\nTest distribution")
    print(test_df["dx"].value_counts())

    print("\nBuilding definitive ResNet50 and loading weights...")
    model = make_resnet50_ham10000()
    model.load_weights(WEIGHTS_PATH)
    test_probabilities, _ = print_model_metrics(model, test_df)

    print("\nLoading cached SHAP metadata and values...")
    with SHAP_PICKLE_PATH.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or not {"shap_values", "idx_eval"}.issubset(payload):
        raise ValueError("Unexpected HAM10000 SHAP pickle format.")

    cached_shap = np.asarray(payload["shap_values"])
    idx_eval = np.asarray(payload["idx_eval"], dtype=np.int64)
    del payload

    expected_shape = (len(idx_eval), IMG_SIZE, IMG_SIZE, 3, 7)
    if cached_shap.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape: {cached_shap.shape}; expected {expected_shape}."
        )
    if np.any(idx_eval < 0) or np.any(idx_eval >= len(test_df)):
        raise IndexError("idx_eval contains indices outside the independent test set.")

    cached_df = test_df.iloc[idx_eval].copy().reset_index(drop=True)
    cached_df["test_index"] = idx_eval
    x_cached = load_images(cached_df)
    y_cached = cached_df["label"].to_numpy(dtype=np.int64)

    print("\nCached local sample")
    print("SHAP shape:", cached_shap.shape)
    print(pd.Series(y_cached).map(CLASS_NAMES).value_counts())

    background_df = train_df.sample(n=BACKGROUND_SIZE, random_state=SEED)
    background = load_images(background_df)

    explainer = SHAPExplainer2(
        model=model,
        x_background=background,
        x_eval=x_cached,
        y_eval=y_cached,
        framework="keras",
        explainer_type="gradient",
        random_state=SEED,
    )
    explainer.shap_values = cached_shap

    cached_predictions = np.argmax(test_probabilities[idx_eval], axis=1)
    local_position = select_local_position(cached_df, cached_predictions)
    local_row = cached_df.iloc[local_position]
    local_test_index = int(local_row["test_index"])
    local_label = CLASS_NAMES[int(local_row["label"])]
    local_prediction = CLASS_NAMES[int(cached_predictions[local_position])]
    instance_label = f"Instance {local_test_index}"

    print("\nSelected local image")
    print("Cache position  :", local_position)
    print("Test index      :", local_test_index)
    print("Image ID        :", local_row["image_id"])
    print("Lesion ID       :", local_row["lesion_id"])
    print("True class      :", local_label)
    print("Predicted class :", local_prediction)

    # Save the selected original image independently.
    original_image_path = (
        RESULT_DIR / f"selected_image_{safe_name(local_row['image_id'])}.png"
    )
    plt.imsave(
        original_image_path,
        np.clip(x_cached[local_position], 0.0, 1.0),
    )
    verify_output(original_image_path, "Original image")

    probability_figure = RESULT_DIR / "probability_differences_local_image.png"
    differences = explainer.plot_probability_differences(
        instance_idx=local_position,
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        top_k=10,
        save_path=probability_figure,
        show=False,
    )
    verify_output(probability_figure, "Probability-difference figure")

    print("\nTop probability differences")
    for rank, (value, class_i, class_j) in enumerate(differences[:10], start=1):
        print(f"{rank:2d}. {class_i} vs {class_j}: |Delta P|={value:.6f}")

    pair_rank = next(
        rank
        for rank, (_, class_i, class_j) in enumerate(differences, start=1)
        if {class_i, class_j} == set(LOCAL_PAIR)
    )
    print(f"\nSelected SHAP-LCD pair: BCC vs BKL (ranking position {pair_rank})")

    explainer.explain_selected_pairs(
        instance_idx=local_position,
        selected_pairs=[LOCAL_PAIR],
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        top_n=10,
        output_dir=RESULT_DIR,
        show=False,
    )
    delta_figure = RESULT_DIR / "deltaS_BCC_vs_BKL.png"
    verify_output(delta_figure, "Delta-S figure")

    heatmap_figure = RESULT_DIR / "heatmap_BCC_vs_BKL.png"
    explainer.plot_shap_heatmap(
        instance_idx=local_position,
        class_i="BCC",
        class_j="BKL",
        class_names=CLASS_NAMES,
        instance_label=instance_label,
        cmap="PiYG",
        alpha=0.60,
        show_original=True,
        save_path=heatmap_figure,
        show=False,
    )
    verify_output(heatmap_figure, "SHAP-LCD heatmap")

    pd.DataFrame(
        [
            {
                "cache_position": local_position,
                "test_index": local_test_index,
                "image_id": local_row["image_id"],
                "lesion_id": local_row["lesion_id"],
                "true_class": local_label,
                "predicted_class": local_prediction,
                "pair": "BCC vs BKL",
                "pair_rank_by_abs_deltaP": pair_rank,
                "original_image_file": original_image_path.name,
            }
        ]
    ).to_csv(RESULT_DIR / "selected_local_image.csv", index=False)

    if not RUN_ROBUSTNESS:
        print("\nLocal SHAP-LCD figures completed.")
        print("Robustness evaluation omitted in this execution.")
        print("Results saved in:", RESULT_DIR)
        return

    # Release the large cached SHAP tensor before recomputing perturbed SHAP.
    explainer.shap_values = None
    del cached_shap, x_cached, y_cached
    gc.collect()

    print("\nPreparing balanced robustness sample...")
    metrics_df = balanced_metric_sample(test_df)
    x_metrics = load_images(metrics_df)
    y_metrics = metrics_df["label"].to_numpy(dtype=np.int64)
    metrics_df[["image_id", "lesion_id", "dx", "label"]].to_csv(
        RESULT_DIR / "robustness_selected_images.csv", index=False
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
        print("\n" + "=" * 76)
        print(
            f"Robustness BCC vs BKL | sigma={configuration['sigma']} | "
            f"tau={configuration['tau']}"
        )
        print("=" * 76)

        instance_results, summary = evaluate_pair_metrics_image(
            model=model,
            shap_explainer=explainer.explainer,
            X_test=x_metrics,
            y_test=y_metrics,
            class_names=CLASS_NAMES,
            class_i_name="BCC",
            class_j_name="BKL",
            filter_mode="true",
            N_max=None,
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
        instance_results.to_csv(
            configuration_dir / "metrics_instances_BCC_vs_BKL.csv", index=False
        )
        summary.to_csv(configuration_dir / "metrics_summary_BCC_vs_BKL.csv")

        for metric_name, row in summary.iterrows():
            all_summaries.append(
                {
                    "configuration": configuration["name"],
                    "sigma": configuration["sigma"],
                    "tau": configuration["tau"],
                    "pair": "BCC vs BKL",
                    "N": len(instance_results),
                    "balance": f"{N_METRICS_PER_CLASS}+{N_METRICS_PER_CLASS}",
                    "M": M_PERTURBATIONS,
                    "metric": metric_name,
                    "mean": row["mean"],
                    "std": row["std"],
                    "median": row["median"],
                    "SE": row["SE"],
                }
            )
        print("\nResults for this configuration")
        print(summary.to_string())

    pd.DataFrame(all_summaries).to_csv(
        RESULT_DIR / "metrics_summary_all_configurations.csv", index=False
    )

    print("\n" + "=" * 76)
    print("HAM10000 SHAP-LCD execution completed")
    print("Results saved in:", RESULT_DIR)
    print("=" * 76)


if __name__ == "__main__":
    main()
