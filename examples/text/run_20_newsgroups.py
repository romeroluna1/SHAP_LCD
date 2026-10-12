"""Reproducible local SHAP-LCD example for 20 Newsgroups."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
MAX_LENGTH = 256
MODEL_BATCH_SIZE = 8
FIXED_PAIR = ("sci.space", "comp.graphics")
TOP_TOKENS = 20
TOP_PROBABILITY_PAIRS = 10

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "20_newsgroups"
MODEL_DIR = ARTIFACT_DIR / "model_20newsgroups_roberta_base"
LABEL_MAPPING_PATH = ARTIFACT_DIR / "label_mapping.csv"
TEST_SPLIT_PATH = ARTIFACT_DIR / "test_split.csv"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "20_newsgroups" / "local"


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_probability_name(class_name: str) -> str:
    return (
        str(class_name)
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def load_class_names(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    required = {"label", "label_name"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{path.name} must contain the columns 'label' and 'label_name'."
        )
    frame = frame.sort_values("label")
    result = {
        int(row.label): str(row.label_name)
        for row in frame.itertuples(index=False)
    }
    if sorted(result) != list(range(len(result))):
        raise ValueError("Class labels must be consecutive from 0 to K-1.")
    return result


def probability_column(frame: pd.DataFrame, class_name: str) -> str:
    candidates = [
        f"prob_{class_name}",
        f"prob_{safe_probability_name(class_name)}",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"No probability column was found for '{class_name}'. "
        f"Expected one of: {candidates}"
    )


def load_saved_test(class_names: dict[int, str]):
    test_frame = pd.read_csv(TEST_SPLIT_PATH)
    prediction_frame = pd.read_csv(TEST_PREDICTIONS_PATH)

    if len(test_frame) != len(prediction_frame):
        raise ValueError(
            "test_split.csv and test_predictions.csv have different lengths: "
            f"{len(test_frame)} vs {len(prediction_frame)}."
        )
    if "text" not in test_frame.columns:
        raise ValueError("test_split.csv must contain a 'text' column.")
    if "pred_label" not in prediction_frame.columns:
        raise ValueError("test_predictions.csv must contain 'pred_label'.")

    if "label" in prediction_frame.columns:
        labels = prediction_frame["label"].to_numpy(dtype=int)
    elif "true_label" in prediction_frame.columns:
        labels = prediction_frame["true_label"].to_numpy(dtype=int)
    elif "label" in test_frame.columns:
        labels = test_frame["label"].to_numpy(dtype=int)
    else:
        raise ValueError("No true-label column was found in the saved files.")

    prediction_columns = [
        probability_column(prediction_frame, class_names[index])
        for index in range(len(class_names))
    ]
    probabilities = prediction_frame[prediction_columns].to_numpy(dtype=float)
    predictions = prediction_frame["pred_label"].to_numpy(dtype=int)
    texts = test_frame["text"].astype(str).tolist()
    return texts, labels, predictions, probabilities


def select_pair_relevant_instance(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_j: int,
    class_r: int,
) -> int:
    """Select deterministically the pair case maximizing min(Pj, Pr)."""
    candidates = np.where(np.isin(predictions, [class_j, class_r]))[0]
    if candidates.size == 0:
        raise ValueError("No test instances were predicted as either pair class.")
    scores = np.minimum(
        probabilities[candidates, class_j],
        probabilities[candidates, class_r],
    )
    return int(candidates[int(np.argmax(scores))])


def truncate_for_shap(text: str, tokenizer) -> tuple[str, int, int]:
    """Explicitly restrict the local document to MAX_LENGTH model tokens."""
    full = tokenizer(text, add_special_tokens=True, truncation=False)
    truncated = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    prepared = tokenizer.decode(
        truncated["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return prepared, len(full["input_ids"]), len(truncated["input_ids"])


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        MODEL_DIR,
        LABEL_MAPPING_PATH,
        TEST_SPLIT_PATH,
        TEST_PREDICTIONS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 20 Newsgroups artifacts:\n - " + "\n - ".join(missing)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = load_class_names(LABEL_MAPPING_PATH)
    inverse_names = {name: index for index, name in class_names.items()}
    if FIXED_PAIR[0] not in inverse_names or FIXED_PAIR[1] not in inverse_names:
        raise ValueError(f"The fixed pair is absent from label_mapping.csv: {FIXED_PAIR}")

    print("=" * 76)
    print(" SHAP-LCD - 20 NEWSGROUPS - LOCAL TEXT EXAMPLE")
    print("=" * 76)
    print(f"Device          : {device}")
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Model           : {MODEL_DIR}")
    print(f"Fixed pair      : {FIXED_PAIR[0]} vs {FIXED_PAIR[1]}")
    print(f"Maximum length  : {MAX_LENGTH}")
    print(f"Results         : {RESULTS_DIR}")
    print("=" * 76)

    print("\nLoading saved test partition and predictions...")
    texts, true_labels, saved_predictions, saved_probabilities = (
        load_saved_test(class_names)
    )

    print("\nLoading trained RoBERTa model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    def predict_proba(input_texts) -> np.ndarray:
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        input_texts = [str(item) for item in list(input_texts)]
        outputs = []
        for start in range(0, len(input_texts), MODEL_BATCH_SIZE):
            batch = input_texts[start:start + MODEL_BATCH_SIZE]
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
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(outputs)

    class_j = inverse_names[FIXED_PAIR[0]]
    class_r = inverse_names[FIXED_PAIR[1]]
    test_index = select_pair_relevant_instance(
        saved_predictions,
        saved_probabilities,
        class_j,
        class_r,
    )
    original_text = texts[test_index]
    text, original_length, prepared_length = truncate_for_shap(
        original_text, tokenizer
    )

    probabilities = predict_proba([text])[0]
    prediction = int(np.argmax(probabilities))

    print("\nSelected test document")
    print(f"Index                   : {test_index}")
    print(f"Selection criterion     : max min(Pj, Pr)")
    print(f"True class              : {class_names[int(true_labels[test_index])]}")
    print(f"Saved predicted class   : {class_names[int(saved_predictions[test_index])]}")
    print(f"Current predicted class : {class_names[prediction]}")
    print(f"Original token count    : {original_length}")
    print(f"Prepared token count    : {prepared_length}")
    print(f"P({FIXED_PAIR[0]}) = {probabilities[class_j]:.6f}")
    print(f"P({FIXED_PAIR[1]}) = {probabilities[class_r]:.6f}")
    print("Text preview            :", text[:350].replace("\n", " "))

    pd.DataFrame(
        {
            "class_index": list(class_names),
            "class_name": [class_names[i] for i in class_names],
            "probability": probabilities,
        }
    ).to_csv(RESULTS_DIR / "selected_document_probabilities.csv", index=False)

    pd.DataFrame(
        [{
            "test_index": test_index,
            "selection_criterion": "max_min_pair_probability",
            "true_class_index": int(true_labels[test_index]),
            "true_class": class_names[int(true_labels[test_index])],
            "predicted_class_index": prediction,
            "predicted_class": class_names[prediction],
            "original_token_count": original_length,
            "prepared_token_count": prepared_length,
            "original_text": original_text,
            "prepared_text": text,
        }]
    ).to_csv(RESULTS_DIR / "selected_document.csv", index=False)

    explainer = TextSHAPLCDExplainer(
        predict_proba=predict_proba,
        tokenizer=tokenizer,
        class_names=class_names,
        output_dir=RESULTS_DIR,
    )
    instance_label = f"test index {test_index}"

    print("\nAlgorithm 1 - probability differences |Delta P|")
    ranking = explainer.plot_probability_differences(
        text=text,
        instance_label=instance_label,
        filename="probability_differences.png",
        top_k=TOP_PROBABILITY_PAIRS,
    )
    print(ranking.head(10).to_string(index=False))

    print("\nAlgorithm 2 - local token differences Delta S")
    token_table = explainer.explain_pair(
        text=text,
        class_j=FIXED_PAIR[0],
        class_r=FIXED_PAIR[1],
        instance_label=instance_label,
        top_n=TOP_TOKENS,
        batch_size=MODEL_BATCH_SIZE,
    )

    print("\nAlgorithm 3 - highlighted SHAP-LCD text")
    html_path = explainer.save_text_highlight(
        text=text,
        token_table=token_table,
        class_j=FIXED_PAIR[0],
        class_r=FIXED_PAIR[1],
        instance_label=instance_label,
    )
    print(f"Highlighted text : {html_path}")

    print("\n" + "=" * 76)
    print(" 20 NEWSGROUPS LOCAL SHAP-LCD EXECUTION COMPLETED")
    print("=" * 76)
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
