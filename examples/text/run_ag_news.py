"""Reproducible local SHAP-LCD example for AG News."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_lcd.text import TextSHAPLCDExplainer  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------
SEED = 42
MAX_LENGTH = 128
MODEL_BATCH_SIZE = 8

# Official AG News test index. Change it to explain another test document.
INSTANCE_INDEX = 9

# Reproducible pair used in the local text illustration.
FIXED_PAIR = ("World", "Sci/Tech")
TOP_TOKENS = 12

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ag_news"
    / "model_agnews_distilbert_base_uncased"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "ag_news" / "local"


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(example: dict) -> dict:
    """Create the same title + description representation used in training."""
    example["text"] = f"{str(example['title'])}. {str(example['description'])}"
    return example


def normalise_labels(dataset):
    labels = np.asarray(dataset["train"]["label"], dtype=int)
    if labels.min() == 1 and labels.max() == 4:
        print("Normalising AG News labels from 1-4 to 0-3...")

        def shift_label(example: dict) -> dict:
            example["label"] = int(example["label"]) - 1
            return example

        return dataset.map(shift_label)

    if labels.min() == 0 and labels.max() == 3:
        print("AG News labels are already encoded from 0 to 3.")
        return dataset

    raise ValueError(
        "Unexpected AG News label range: "
        f"minimum={labels.min()}, maximum={labels.max()}."
    )


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            "The trained AG News model directory was not found:\n"
            f"{MODEL_DIR}"
        )

    required = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "tokenizer_config.json",
    ]
    if not ((MODEL_DIR / "model.safetensors").exists() or
            (MODEL_DIR / "pytorch_model.bin").exists()):
        required.append(MODEL_DIR / "model.safetensors")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n - " + "\n - ".join(missing)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(" SHAP-LCD - AG NEWS - LOCAL TEXT EXAMPLE")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Model           : {MODEL_DIR}")
    print(f"Test index      : {INSTANCE_INDEX}")
    print(f"Fixed pair      : {FIXED_PAIR[0]} vs {FIXED_PAIR[1]}")
    print(f"Results         : {RESULTS_DIR}")
    print("=" * 72)

    print("\nLoading trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    print("\nLoading the official AG News test set...")
    dataset = load_dataset("sh0416/ag_news")
    dataset = dataset.map(build_text)
    dataset = normalise_labels(dataset)
    test_dataset = dataset["test"]

    if not 0 <= INSTANCE_INDEX < len(test_dataset):
        raise IndexError(
            f"INSTANCE_INDEX={INSTANCE_INDEX} is outside the test range "
            f"0-{len(test_dataset) - 1}."
        )

    selected = test_dataset[int(INSTANCE_INDEX)]
    text = str(selected["text"])
    true_index = int(selected["label"])
    true_name = CLASS_NAMES[true_index]

    def predict_proba(texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(item) for item in list(texts)]
        chunks = []
        for start in range(0, len(texts), MODEL_BATCH_SIZE):
            batch = texts[start:start + MODEL_BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(chunks)

    probs = predict_proba([text])[0]
    predicted_index = int(np.argmax(probs))
    predicted_name = CLASS_NAMES[predicted_index]

    print("\nSelected test document")
    print(f"Index           : {INSTANCE_INDEX}")
    print(f"True class      : {true_name}")
    print(f"Predicted class : {predicted_name}")
    print("Text preview    :", text[:300].replace("\n", " "))

    probability_table = pd.DataFrame(
        {
            "class_index": list(CLASS_NAMES),
            "class_name": [CLASS_NAMES[i] for i in CLASS_NAMES],
            "probability": probs,
        }
    )
    probability_table.to_csv(
        RESULTS_DIR / "selected_document_probabilities.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "test_index": INSTANCE_INDEX,
                "true_class_index": true_index,
                "true_class": true_name,
                "predicted_class_index": predicted_index,
                "predicted_class": predicted_name,
                "text": text,
            }
        ]
    ).to_csv(RESULTS_DIR / "selected_document.csv", index=False)

    explainer = TextSHAPLCDExplainer(
        predict_proba=predict_proba,
        tokenizer=tokenizer,
        class_names=CLASS_NAMES,
        output_dir=RESULTS_DIR,
        extra_plot_stopwords={"first", "two"},
    )

    instance_label = f"test index {INSTANCE_INDEX}"

    print("\nAlgorithm 1 - probability differences |Delta P|")
    ranking = explainer.plot_probability_differences(
        text=text,
        instance_label=instance_label,
        filename="probability_differences.png",
    )
    print(ranking.to_string(index=False))

    inverse_names = {name: idx for idx, name in CLASS_NAMES.items()}
    class_j = inverse_names[FIXED_PAIR[0]]
    class_r = inverse_names[FIXED_PAIR[1]]
    pair_row = ranking[
        ((ranking["class_j_index"] == class_j)
         & (ranking["class_r_index"] == class_r))
        | ((ranking["class_j_index"] == class_r)
           & (ranking["class_r_index"] == class_j))
    ].iloc[0]
    print(
        "\nSelected pair: "
        f"{CLASS_NAMES[class_j]} vs {CLASS_NAMES[class_r]} "
        f"(|Delta P|={pair_row['abs_delta_probability']:.6f})"
    )

    print("\nAlgorithm 2 - local token differences Delta S")
    token_table = explainer.explain_pair(
        text=text,
        class_j=class_j,
        class_r=class_r,
        instance_label=instance_label,
        top_n=TOP_TOKENS,
        batch_size=MODEL_BATCH_SIZE,
    )

    print("\nAlgorithm 3 - highlighted SHAP-LCD text")
    html_path = explainer.save_text_highlight(
        text=text,
        token_table=token_table,
        class_j=class_j,
        class_r=class_r,
        instance_label=instance_label,
    )
    print(f"Highlighted text : {html_path}")

    print("\n" + "=" * 72)
    print(" AG NEWS LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
