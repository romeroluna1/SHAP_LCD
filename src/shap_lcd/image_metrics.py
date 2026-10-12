"""Stability and consistency metrics for SHAP-LCD image explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    denominator = (np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)
    return float(np.dot(a, b) / denominator)


def _topk_and_l2(vector: np.ndarray, top_k: int | None, eps: float = 1e-12):
    """Keep the top-k absolute attributions and apply L2 normalization."""
    values = np.asarray(vector).reshape(-1).copy()
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k debe ser positivo o None.")
        if top_k < values.size:
            indices = np.argpartition(np.abs(values), -top_k)[-top_k:]
            mask = np.zeros_like(values, dtype=bool)
            mask[indices] = True
            values[~mask] = 0.0
    return values / (np.linalg.norm(values) + eps)


def _class_index(class_name: str, class_names: dict[int, str]) -> int:
    inverse = {name: index for index, name in class_names.items()}
    if class_name not in inverse:
        raise ValueError(f"Clase desconocida: {class_name}")
    return inverse[class_name]


def _delta_s_from_shap(shap_values, class_i: int, class_j: int) -> np.ndarray:
    """Return a batch of pairwise differential SHAP maps."""
    if isinstance(shap_values, list):
        difference = np.asarray(shap_values[class_i]) - np.asarray(shap_values[class_j])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 5:  # (N,H,W,C,K)
            difference = values[..., class_i] - values[..., class_j]
        elif values.ndim == 4:  # (N,H,W,K)
            difference = values[..., class_i] - values[..., class_j]
        else:
            raise ValueError(f"Formato SHAP no reconocido: {values.shape}")

    difference = np.asarray(difference)

    # Agregar los canales de color para obtener un mapa diferencial por
    # posición espacial. La suma conserva la aditividad de SHAP.
    if difference.ndim == 4:
        difference = np.sum(difference, axis=-1)

    if difference.ndim != 3:
        raise ValueError(
            "Se esperaba un lote de mapas SHAP con forma (N,H,W) "
            "después de agregar los canales; "
            f"se recibió {difference.shape}."
        )

    return difference


def evaluate_instance_metrics_image(
    model,
    shap_explainer,
    x: np.ndarray,
    class_i: int,
    class_j: int,
    M_pert: int = 40,
    sigma: float = 0.02,
    tau: float = 0.90,
    top_k: int | None = None,
    seed: int = 42,
):
    """Evaluate SHAP-LCD robustness metrics for one image."""
    if M_pert < 1:
        raise ValueError("M_pert debe ser al menos 1.")
    if sigma < 0:
        raise ValueError("sigma no puede ser negativo.")
    if not 0 <= tau <= 1:
        raise ValueError("tau debe pertenecer al intervalo [0,1].")

    x = np.asarray(x, dtype=np.float32)
    random_generator = np.random.default_rng(seed)

    probabilities = model.predict(x[None, ...], verbose=0)[0]
    prediction_base = int(np.argmax(probabilities))
    delta_p = float(abs(probabilities[class_i] - probabilities[class_j]))

    noise = random_generator.normal(
        loc=0.0,
        scale=sigma,
        size=(M_pert,) + x.shape,
    ).astype(np.float32)
    x_perturbed = np.clip(x[None, ...] + noise, 0.0, 1.0).astype(np.float32)

    # The base image and all perturbations are explained in one batch.
    x_all = np.concatenate([x[None, ...], x_perturbed], axis=0)
    shap_all = shap_explainer.shap_values(x_all)
    delta_s_all = _delta_s_from_shap(shap_all, class_i, class_j)

    explanation_base = _topk_and_l2(delta_s_all[0], top_k)
    explanations_perturbed = np.stack(
        [_topk_and_l2(delta_s_all[m], top_k) for m in range(1, M_pert + 1)],
        axis=0,
    )

    differences = explanations_perturbed - explanation_base[None, :]
    stability = float(np.mean(np.sum(differences * differences, axis=1)))

    probabilities_perturbed = model.predict(x_perturbed, verbose=0)
    predictions_perturbed = np.argmax(probabilities_perturbed, axis=1)
    prediction_kept = predictions_perturbed == prediction_base

    similarities = np.asarray(
        [
            _cosine_similarity(explanation_base, explanations_perturbed[m])
            for m in range(M_pert)
        ],
        dtype=float,
    )

    kept_count = int(np.sum(prediction_kept))
    if kept_count == 0:
        consistency_predexp = np.nan
    else:
        consistency_predexp = float(
            np.mean(similarities[prediction_kept] >= tau)
        )

    similar_explanation = similarities >= tau
    consistency_imp = float(np.mean((~similar_explanation) | prediction_kept))
    coverage_imp = float(np.mean(similar_explanation))
    prediction_keep_imp = float(np.mean(prediction_kept))

    return {
        "deltaP_ij": delta_p,
        "pred_base": prediction_base,
        "Stability_S": stability,
        "Consistency_C_predexp": consistency_predexp,
        "Consistency_C_imp": consistency_imp,
        "Coverage_imp": coverage_imp,
        "PredKeep_imp": prediction_keep_imp,
        "M_keep": kept_count,
    }


def evaluate_pair_metrics_image(
    model,
    shap_explainer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: dict[int, str],
    class_i_name: str,
    class_j_name: str,
    filter_mode: str = "true",
    N_max: int | None = None,
    M_pert: int = 40,
    sigma: float = 0.02,
    tau: float = 0.90,
    top_k: int | None = None,
    seed: int = 42,
):
    """Evaluate image robustness metrics for a selected class pair."""
    class_i = _class_index(class_i_name, class_names)
    class_j = _class_index(class_j_name, class_names)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    if len(X_test) != len(y_test):
        raise ValueError("X_test y y_test deben tener el mismo número de instancias.")
    if filter_mode not in {"true", "all"}:
        raise ValueError("filter_mode debe ser 'true' o 'all'.")

    if filter_mode == "true":
        indices = np.where((y_test == class_i) | (y_test == class_j))[0]
    else:
        indices = np.arange(len(X_test))

    if N_max is not None:
        if N_max < 1:
            raise ValueError("N_max debe ser positivo o None.")
        indices = indices[:N_max]
    if len(indices) == 0:
        raise ValueError("No existen instancias para el par seleccionado.")

    rows = []
    for position, index in enumerate(indices, start=1):
        result = evaluate_instance_metrics_image(
            model=model,
            shap_explainer=shap_explainer,
            x=X_test[index],
            class_i=class_i,
            class_j=class_j,
            M_pert=M_pert,
            sigma=sigma,
            tau=tau,
            top_k=top_k,
            # Preserve the exact seed rule of the original experiment.
            seed=seed + int(index),
        )
        rows.append(
            {
                "idx_test": int(index),
                "true_label": int(y_test[index]),
                "pair": f"{class_i_name} vs {class_j_name}",
                **result,
            }
        )
        print(
            f"[{position:02d}/{len(indices)}] idx_test={index:5d} | "
            f"S={result['Stability_S']:.6f} | "
            f"Cimp={result['Consistency_C_imp']:.6f} | "
            f"Cov={result['Coverage_imp']:.6f} | "
            f"PredKeep={result['PredKeep_imp']:.6f}"
        )

    instance_results = pd.DataFrame(rows)
    metric_columns = [
        "Stability_S",
        "Consistency_C_predexp",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]
    summary = instance_results[metric_columns].agg(["mean", "std"])
    return instance_results, summary


__all__ = [
    "evaluate_instance_metrics_image",
    "evaluate_pair_metrics_image",
]
