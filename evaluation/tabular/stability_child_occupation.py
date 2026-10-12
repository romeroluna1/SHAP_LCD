# ============================================================
# main_valid_perturbations_shap_lcd_4class.py
# ============================================================
# Recalcula estabilidad y consistencia de SHAP-LCD usando
# perturbaciones VÁLIDAS en el ESPACIO ORIGINAL de variables.
#
# Objetivo:
# - responder a la observación del revisor sobre perturbaciones
#   de variables binarias/categóricas;
# - evitar valores fraccionarios inválidos en variables one-hot
#   o binarias generados por ruido gaussiano en espacio procesado.
#
# Estrategia:
#   * Numéricas continuas:
#       ruido gaussiano local, con sigma relativo al rango observado
#       en entrenamiento, y recorte a [min_train, max_train].
#   * Numéricas discretas/enteras:
#       ruido local + proyección al valor observado más cercano.
#   * Binarias:
#       se mantienen en sus valores válidos observados; con
#       probabilidad P_CHANGE se cambia al otro valor válido.
#   * Categóricas/ordinales:
#       con probabilidad P_CHANGE se reemplazan por otra categoría
#       válida observada en entrenamiento, muestreada según su
#       frecuencia empírica.
#
# Después de perturbar en espacio original:
#   -> misma selección de variables ya ajustada
#   -> mismo encoder ya ajustado
#   -> mismo MinMaxScaler ya ajustado
#   -> mismo Random Forest final
#   -> SHAP / SHAP-LCD
#
# NO reentrena el modelo.
# ============================================================

import json
import sys
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# Required to deserialize processing_selection_pipeline.joblib. Keep a copy of
# preprocessing_selection.py next to this evaluation script.
from preprocessing_selection import ProcessingSelectionPipeline  # noqa: F401


# ============================================================
# CLASE DEL MODELO ORIGINAL
# ============================================================
# El modelo guardado por el main principal fue serializado cuando
# esta clase estaba definida dentro de __main__. Para poder cargar
# exactamente ese artefacto desde este script independiente, la
# clase debe existir con el mismo nombre antes de joblib.load().

class MyOptimizedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, random_state=42):
        super().__init__(
            n_estimators=800,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features=0.5,
            class_weight=None,
            n_jobs=-1,
            random_state=random_state
        )


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "child_occupation"

# Use True first to verify the environment. Change to False for the definitive
# reproduction of the validated editorial experiment.
QUICK_TEST = True
RUN_TAG = "quick_test" if QUICK_TEST else "final"
OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "child_occupation"
    / RUN_TAG
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Definitive protocol: N=50 and M=30. Quick test: N=2 and M=2.
N_INSTANCES = 2 if QUICK_TEST else 50
M_PERTURB = 2 if QUICK_TEST else 30
SIGMA = 0.005
TAU = 0.85
TOP_K = 10
RANDOM_STATE = 42

# Probabilidad local de modificar una variable no numérica.
# 0.05 = 5 % por variable y perturbación.
P_CHANGE = 0.05

CLASS_NAMES = {
    0: "asoTrabajo",
    1: "estudia",
    2: "incapacitado",
    3: "otraactividad",
}

PAIRS = [
    (0, 1),  # asoTrabajo vs estudia
    (0, 3),  # asoTrabajo vs otraactividad
]


# ============================================================
# HELPERS
# ============================================================

def _drop_saved_index_columns(df):
    """Elimina columnas auxiliares creadas al guardar CSV con index=True."""
    cols_to_drop = [
        c for c in df.columns
        if str(c).startswith("Unnamed:")
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(
        np.dot(a, b)
        / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
    )


def normalize_l2(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def top_k_sparse(v, top_k=10):
    v = np.asarray(v, dtype=float)
    if top_k is None or top_k >= len(v):
        return v
    idx = np.argsort(np.abs(v))[::-1][:top_k]
    out = np.zeros_like(v)
    out[idx] = v[idx]
    return out


def shap_lcd_vector(tree_explainer, x_processed, class_i, class_j, top_k=10):
    """
    Calcula ΔS = SHAP(C_i) - SHAP(C_j), top-k y normalización L2.
    """
    x_processed = np.asarray(x_processed, dtype=float).reshape(1, -1)
    shap_raw = tree_explainer.shap_values(x_processed)

    if isinstance(shap_raw, list):
        phi_i = np.asarray(shap_raw[class_i][0], dtype=float)
        phi_j = np.asarray(shap_raw[class_j][0], dtype=float)
    else:
        arr = np.asarray(shap_raw)

        # Formato habitual actual:
        # (n_instances, n_features, n_classes)
        if arr.ndim == 3 and arr.shape[0] == 1:
            phi_i = np.asarray(arr[0, :, class_i], dtype=float)
            phi_j = np.asarray(arr[0, :, class_j], dtype=float)
        elif arr.ndim == 2:
            # fallback poco frecuente
            phi_i = np.asarray(arr[class_i], dtype=float)
            phi_j = np.asarray(arr[class_j], dtype=float)
        else:
            raise ValueError(
                f"Formato SHAP no reconocido para una instancia: {arr.shape}"
            )

    v = phi_i - phi_j
    v = top_k_sparse(v, top_k=top_k)
    v = normalize_l2(v)
    return v


def transform_raw_rows(raw_df, processor, scaler):
    """
    Aplica exactamente los objetos ya ajustados durante entrenamiento.
    """
    selected_cols = processor.get_selected_original_features()

    missing = [c for c in selected_cols if c not in raw_df.columns]
    if missing:
        raise KeyError(
            "Faltan variables seleccionadas en los datos raw:\n"
            + "\n".join(missing)
        )

    raw_sel = raw_df[selected_cols].copy()
    x_proc = processor.preprocessor_.transform(raw_sel)
    x_final = scaler.transform(x_proc)

    return np.asarray(x_final, dtype=float)


def infer_feature_types(processor, selected_cols, X_train_raw):
    """
    Recupera los tipos aprendidos por ProcessingSelectionPipeline.
    Si alguna variable no aparece, aplica una inferencia conservadora.
    """
    learned = getattr(processor, "feature_types_", {}) or {}
    result = {}

    for col in selected_cols:
        if col in learned:
            result[col] = str(learned[col])
            continue

        s = X_train_raw[col].dropna()
        nunique = s.nunique()

        if nunique <= 2:
            result[col] = "binary"
        elif pd.api.types.is_numeric_dtype(s):
            result[col] = "numeric"
        else:
            result[col] = "categorical"

    return result


def build_training_support(X_train_raw, selected_cols, feature_types):
    """
    Construye el soporte válido observado en entrenamiento.
    """
    support = {}

    for col in selected_cols:
        s = X_train_raw[col].dropna()
        ftype = feature_types[col]

        if len(s) == 0:
            support[col] = {"type": ftype}
            continue

        if ftype == "numeric":
            vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
            uniq = np.unique(vals)

            if len(vals) == 0:
                support[col] = {"type": ftype}
                continue

            is_integer_like = bool(
                np.all(np.isclose(vals, np.round(vals), atol=1e-10))
            )

            support[col] = {
                "type": ftype,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "is_integer_like": is_integer_like,
                "observed_values": uniq,
            }

        else:
            vc = s.value_counts(normalize=True, dropna=True)
            support[col] = {
                "type": ftype,
                "values": vc.index.to_numpy(dtype=object),
                "probs": vc.to_numpy(dtype=float),
            }

    return support


def _sample_alternative(current, values, probs, rng):
    """
    Muestrea una categoría válida distinta de la actual cuando sea posible.
    """
    values = np.asarray(values, dtype=object)
    probs = np.asarray(probs, dtype=float)

    if len(values) <= 1:
        return current

    mask = np.array([str(v) != str(current) for v in values], dtype=bool)

    if not np.any(mask):
        return current

    alt_values = values[mask]
    alt_probs = probs[mask]
    alt_probs = alt_probs / alt_probs.sum()

    return rng.choice(alt_values, p=alt_probs)


def perturb_raw_row(
    base_row,
    selected_cols,
    feature_types,
    support,
    rng,
    sigma=0.005,
    p_change=0.05,
):
    """
    Genera UNA perturbación válida en el espacio original.
    """
    row = base_row.copy()

    for col in selected_cols:
        ftype = feature_types[col]
        info = support[col]

        if pd.isna(row[col]):
            # No introducimos nuevos NA.
            continue

        if ftype == "numeric":
            try:
                current = float(row[col])
            except Exception:
                continue

            value_range = float(info.get("range", 0.0))

            if value_range <= 0:
                continue

            pert = current + rng.normal(
                loc=0.0,
                scale=sigma * value_range,
            )

            pert = float(
                np.clip(
                    pert,
                    info["min"],
                    info["max"],
                )
            )

            # Variables numéricas originalmente enteras:
            # proyectar al valor observado más cercano para no crear
            # edades/cantidades fraccionarias imposibles.
            if info.get("is_integer_like", False):
                observed = np.asarray(
                    info["observed_values"],
                    dtype=float,
                )
                pert = float(
                    observed[np.argmin(np.abs(observed - pert))]
                )

                # conservar dtype/int cuando sea posible
                if np.isclose(pert, round(pert)):
                    pert = int(round(pert))

            row[col] = pert

        elif ftype in {"binary", "categorical", "ordinal"}:
            # La variable queda válida siempre. Solo cambia con
            # probabilidad pequeña.
            if rng.random() < p_change:
                row[col] = _sample_alternative(
                    current=row[col],
                    values=info.get("values", []),
                    probs=info.get("probs", []),
                    rng=rng,
                )

        else:
            # Tipo desconocido: tratamiento conservador.
            # No modificar.
            pass

    return row


def sample_pair_indices(
    y_true,
    class_i,
    class_j,
    n_instances=50,
    seed=42,
):
    """
    Muestreo balanceado por clase verdadera.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)

    idx_i = np.where(y_true == class_i)[0]
    idx_j = np.where(y_true == class_j)[0]

    rng.shuffle(idx_i)
    rng.shuffle(idx_j)

    n = min(n_instances, len(idx_i) + len(idx_j))
    n_i = n // 2
    n_j = n - n_i

    take_i = min(n_i, len(idx_i))
    take_j = min(n_j, len(idx_j))

    remaining = n - (take_i + take_j)

    if remaining > 0:
        extra_i = min(remaining, len(idx_i) - take_i)
        take_i += extra_i
        remaining -= extra_i

    if remaining > 0:
        extra_j = min(remaining, len(idx_j) - take_j)
        take_j += extra_j

    selected = np.concatenate([
        idx_i[:take_i],
        idx_j[:take_j],
    ])

    rng.shuffle(selected)
    return selected


def evaluate_pair_valid_perturbations(
    pair,
    X_train_raw,
    X_test_raw,
    y_test,
    processor,
    scaler,
    model,
    tree_explainer,
    selected_cols,
    feature_types,
    support,
    n_instances=50,
    M=30,
    sigma=0.005,
    p_change=0.05,
    tau=0.85,
    top_k=10,
    seed=42,
):
    class_i, class_j = pair

    idxs = sample_pair_indices(
        y_true=y_test,
        class_i=class_i,
        class_j=class_j,
        n_instances=n_instances,
        seed=seed,
    )

    print("\n" + "=" * 72)
    print(
        f" PAR: {CLASS_NAMES[class_i]} vs {CLASS_NAMES[class_j]}"
    )
    print("=" * 72)
    print("Instancias seleccionadas:", len(idxs))
    print("Distribución real:")
    print(pd.Series(np.asarray(y_test)[idxs]).value_counts().sort_index())

    rows = []

    for t, idx in enumerate(idxs):
        rng = np.random.default_rng(seed + 10000 * (class_i + 1) + 1000 * (class_j + 1) + t)

        base_raw = X_test_raw.iloc[idx][selected_cols].copy()
        base_proc = transform_raw_rows(
            pd.DataFrame([base_raw]),
            processor,
            scaler,
        )[0]

        base_pred = int(
            model.predict(base_proc.reshape(1, -1))[0]
        )

        base_exp = shap_lcd_vector(
            tree_explainer,
            base_proc,
            class_i,
            class_j,
            top_k=top_k,
        )

        stability_values = []
        consistency_values = []
        coverage_values = []
        pred_keep_values = []
        cosine_values = []

        for _ in range(M):
            pert_raw = perturb_raw_row(
                base_row=base_raw,
                selected_cols=selected_cols,
                feature_types=feature_types,
                support=support,
                rng=rng,
                sigma=sigma,
                p_change=p_change,
            )

            pert_proc = transform_raw_rows(
                pd.DataFrame([pert_raw]),
                processor,
                scaler,
            )[0]

            pred_p = int(
                model.predict(pert_proc.reshape(1, -1))[0]
            )

            exp_p = shap_lcd_vector(
                tree_explainer,
                pert_proc,
                class_i,
                class_j,
                top_k=top_k,
            )

            squared_distance = float(
                np.sum((exp_p - base_exp) ** 2)
            )
            sim = cosine_similarity(base_exp, exp_p)

            antecedent = bool(sim >= tau)
            consequent = bool(pred_p == base_pred)
            implication_holds = (not antecedent) or consequent

            stability_values.append(squared_distance)
            cosine_values.append(sim)
            coverage_values.append(int(antecedent))
            pred_keep_values.append(int(consequent))
            consistency_values.append(int(implication_holds))

        p = model.predict_proba(
            base_proc.reshape(1, -1)
        )[0]

        rows.append({
            "idx_test": int(idx),
            "pair": (
                f"{CLASS_NAMES[class_i]} vs "
                f"{CLASS_NAMES[class_j]}"
            ),
            "true_class": int(np.asarray(y_test)[idx]),
            "base_prediction": base_pred,
            "abs_deltaP_ij": float(
                abs(p[class_i] - p[class_j])
            ),
            "Stability_S": float(
                np.mean(stability_values)
            ),
            "Consistency_C_imp": float(
                np.mean(consistency_values)
            ),
            "Coverage_imp": float(
                np.mean(coverage_values)
            ),
            "PredKeep_imp": float(
                np.mean(pred_keep_values)
            ),
            "Cosine_mean": float(
                np.mean(cosine_values)
            ),
        })

        if (t + 1) % 10 == 0 or (t + 1) == len(idxs):
            print(
                f"Procesadas {t + 1}/{len(idxs)} instancias..."
            )

    df = pd.DataFrame(rows)

    metric_cols = [
        "Stability_S",
        "Consistency_C_imp",
        "Coverage_imp",
        "PredKeep_imp",
    ]

    summary = df[metric_cols].agg(["mean", "std"]).T
    summary["mean±std"] = (
        summary["mean"].map(lambda x: f"{x:.6f}")
        + " ± "
        + summary["std"].map(lambda x: f"{x:.6f}")
    )

    print("\nResumen:")
    print(summary[["mean±std"]].to_string())

    return df, summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("SHAP:", shap.__version__)
    print("OS:", platform.platform())
    print("Processor:", platform.processor())

    print("\n" + "=" * 72)
    print(" SHAP-LCD: PERTURBACIONES VÁLIDAS EN ESPACIO ORIGINAL")
    print("=" * 72)
    print("ARTIFACT_DIR :", ARTIFACT_DIR)
    print("QUICK_TEST   :", QUICK_TEST)
    print("RESULTS      :", OUT_DIR)
    print("N_INSTANCES  :", N_INSTANCES)
    print("M_PERTURB    :", M_PERTURB)
    print("SIGMA numeric:", SIGMA)
    print("P_CHANGE cat/bin:", P_CHANGE)
    print("TAU          :", TAU)
    print("TOP_K        :", TOP_K)

    required = {
        "X_train_raw": ARTIFACT_DIR / "X_train_raw_before_processing.csv",
        "X_test_raw": ARTIFACT_DIR / "X_test_raw_before_processing.csv",
        "y_test": ARTIFACT_DIR / "y_test_original_split.csv",
        "processor": ARTIFACT_DIR / "processing_selection_pipeline.joblib",
        "scaler": ARTIFACT_DIR / "minmax_scaler.joblib",
        "model": ARTIFACT_DIR / "random_forest_optimizado.joblib",
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "No se encontraron estos artefactos:\n"
            + "\n".join(missing)
            + "\n\nCopia el paquete definitivo dentro de "
              "artifacts/child_occupation."
        )

    X_train_raw = _drop_saved_index_columns(
        pd.read_csv(required["X_train_raw"])
    )
    X_test_raw = _drop_saved_index_columns(
        pd.read_csv(required["X_test_raw"])
    )

    y_test_df = pd.read_csv(required["y_test"])
    y_test = y_test_df.iloc[:, 0].to_numpy(dtype=int)

    processor = joblib.load(required["processor"])
    scaler = joblib.load(required["scaler"])
    model = joblib.load(required["model"])

    selected_cols = list(
        processor.get_selected_original_features()
    )

    feature_types = infer_feature_types(
        processor=processor,
        selected_cols=selected_cols,
        X_train_raw=X_train_raw,
    )

    support = build_training_support(
        X_train_raw=X_train_raw,
        selected_cols=selected_cols,
        feature_types=feature_types,
    )

    print("\nVariables originales seleccionadas:", len(selected_cols))
    print("\nTipos de variables seleccionadas:")
    print(pd.Series(feature_types).value_counts())

    # Guardar reporte del esquema usado para perturbar.
    pd.DataFrame({
        "feature": selected_cols,
        "type": [feature_types[c] for c in selected_cols],
    }).to_csv(
        OUT_DIR / "valid_perturbation_feature_types.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Verificación de soporte no numérico.
    support_rows = []
    for col in selected_cols:
        info = support[col]
        if info["type"] in {"binary", "categorical", "ordinal"}:
            support_rows.append({
                "feature": col,
                "type": info["type"],
                "valid_values": " | ".join(
                    map(str, info.get("values", []))
                ),
            })

    pd.DataFrame(support_rows).to_csv(
        OUT_DIR / "valid_categories_and_binary_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tree_explainer = shap.TreeExplainer(model)

    all_summaries = []

    for pair in PAIRS:
        df_pair, summary = evaluate_pair_valid_perturbations(
            pair=pair,
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            y_test=y_test,
            processor=processor,
            scaler=scaler,
            model=model,
            tree_explainer=tree_explainer,
            selected_cols=selected_cols,
            feature_types=feature_types,
            support=support,
            n_instances=N_INSTANCES,
            M=M_PERTURB,
            sigma=SIGMA,
            p_change=P_CHANGE,
            tau=TAU,
            top_k=TOP_K,
            seed=RANDOM_STATE,
        )

        pair_tag = (
            f"{CLASS_NAMES[pair[0]]}_vs_"
            f"{CLASS_NAMES[pair[1]]}"
        )

        df_pair.to_csv(
            OUT_DIR / f"valid_perturbations_instance_metrics_{pair_tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_out = summary.reset_index().rename(
            columns={"index": "metric"}
        )
        summary_out.insert(0, "pair", pair_tag)

        summary_out.to_csv(
            OUT_DIR / f"valid_perturbations_summary_{pair_tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_summaries.append(summary_out)

    df_all_summary = pd.concat(
        all_summaries,
        axis=0,
        ignore_index=True,
    )

    df_all_summary.to_csv(
        OUT_DIR / "valid_perturbations_summary_all_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "protocol": "valid perturbations in original feature space",
        "quick_test": QUICK_TEST,
        "n_instances_per_pair": N_INSTANCES,
        "M_perturbations_per_instance": M_PERTURB,
        "sigma_numeric_relative_to_training_range": SIGMA,
        "p_change_binary_categorical_ordinal": P_CHANGE,
        "tau": TAU,
        "top_k": TOP_K,
        "random_state": RANDOM_STATE,
        "numeric_continuous": (
            "bounded Gaussian perturbation in original scale; "
            "clipped to observed training min/max"
        ),
        "numeric_integer_like": (
            "bounded Gaussian perturbation followed by projection "
            "to nearest value observed in training"
        ),
        "binary": (
            "kept in observed valid states; alternative valid state "
            "sampled with probability p_change"
        ),
        "categorical_ordinal": (
            "alternative observed category sampled from training "
            "empirical distribution with probability p_change"
        ),
        "transformation": (
            "perturbed raw rows transformed using the same fitted "
            "processor/encoder and MinMaxScaler as the final model"
        ),
    }

    with open(
        OUT_DIR / "valid_perturbations_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 72)
    print(" RESULTADOS FINALES")
    print("=" * 72)
    print(df_all_summary[
        ["pair", "metric", "mean±std"]
    ].to_string(index=False))

    print("\nArchivos guardados en:")
    print(OUT_DIR)

    print("\nProceso finalizado correctamente.")
    if QUICK_TEST:
        print("Para la reproducción definitiva, usa QUICK_TEST = False.")


if __name__ == "__main__":
    main()
