"""SHAP-LCD utilities for image classifiers implemented with Keras."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


class SHAPExplainer2:
    """Compute and visualize local pairwise SHAP differences for images."""

    def __init__(
            self,
            model,
            x_background,
            x_eval,
            y_eval=None,
            framework="keras",
            explainer_type="deep",
            random_state=42,
    ):
        self.model = model
        self.framework = framework
        self.explainer_type = str(explainer_type).lower()
        self.random_state = random_state

        self.x_background = self._ensure_4d_float(x_background)
        self.x_eval = self._ensure_4d_float(x_eval)
        self.y_eval = None if y_eval is None else np.asarray(y_eval)

        if self.y_eval is not None and len(self.y_eval) != len(self.x_eval):
            raise ValueError("x_eval y y_eval deben tener el mismo número de instancias.")

        self.explainer = None
        self.shap_values = None
        self.max_eval_used = None
        self.model_hash = self._fingerprint_model(self.model)
        self.background_hash = self._fingerprint_array(self.x_background)
        self._loaded_meta = None

        self._init_explainer()

    @staticmethod
    def _ensure_4d_float(x):
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[..., np.newaxis]
        if x.ndim != 4:
            raise ValueError(f"Se esperaba (N,H,W,C). Forma recibida: {x.shape}")
        x = x.astype(np.float32)
        if x.size and x.max() > 1.0:
            x = x / 255.0
        return x

    @staticmethod
    def _fingerprint_array(arr):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(arr).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _fingerprint_model(model):
        digest = hashlib.sha256()
        for weight in model.get_weights():
            digest.update(np.ascontiguousarray(weight).tobytes())
        return digest.hexdigest()

    def _init_explainer(self):
        if self.framework != "keras":
            raise ValueError("Esta implementación requiere framework='keras'.")

        if self.explainer_type == "deep":
            self.explainer = shap.DeepExplainer(
                self.model,
                self.x_background,
            )
        elif self.explainer_type == "gradient":
            self.explainer = shap.GradientExplainer(
                self.model,
                self.x_background,
            )
        else:
            raise ValueError(
                "explainer_type debe ser 'deep' o 'gradient'."
            )

    def _get_eval_slice(self, max_eval):
        if max_eval is None:
            return self.x_eval
        max_eval = int(max_eval)
        if max_eval < 1 or max_eval > len(self.x_eval):
            raise ValueError(
                f"max_eval debe estar entre 1 y {len(self.x_eval)}; se recibió {max_eval}."
            )
        return self.x_eval[:max_eval]

    def is_cache_compatible(self, desired_max_eval):
        """Validate cached SHAP values against model, background and sample size."""
        if self._loaded_meta is None:
            return False, "no hay metadata cargada"
        if self._loaded_meta.get("model_hash") != self.model_hash:
            return False, "modelo cambió (model_hash distinto)"
        if self._loaded_meta.get("background_hash") != self.background_hash:
            return False, "background cambió (background_hash distinto)"
        if self._loaded_meta.get("max_eval_used") != desired_max_eval:
            return False, "MAX_EVAL cambió (max_eval_used distinto)"
        return True, "ok"

    def calculate_shap_values(self, max_eval=None):
        x_selected = self._get_eval_slice(max_eval)
        self.max_eval_used = None if max_eval is None else int(max_eval)
        self.shap_values = self.explainer.shap_values(x_selected)
        print("Valores SHAP calculados. Forma:", self._describe_shap_shape())

    def save_shap_values(self, filename="shap_values.pickle"):
        if self.shap_values is None:
            raise ValueError("No hay valores SHAP para guardar.")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "shap_values": self.shap_values,
            "max_eval_used": self.max_eval_used,
            "model_hash": self.model_hash,
            "background_hash": self.background_hash,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_shap_values(self, filename="shap_values.pickle"):
        path = Path(filename)
        with path.open("rb") as file:
            payload = pickle.load(file)

        if isinstance(payload, dict) and "shap_values" in payload:
            self.shap_values = payload["shap_values"]
            self.max_eval_used = payload.get("max_eval_used")
            self._loaded_meta = payload
        else:
            # Backward compatibility with legacy pickles containing only SHAP values.
            self.shap_values = payload
            self.max_eval_used = None
            self._loaded_meta = {
                "max_eval_used": None,
                "model_hash": None,
                "background_hash": None,
            }
        print("Valores SHAP cargados. Forma:", self._describe_shap_shape())

    def _describe_shap_shape(self):
        if isinstance(self.shap_values, list):
            return [np.asarray(values).shape for values in self.shap_values]
        return np.asarray(self.shap_values).shape

    def _values_for_class(self, instance_idx, class_idx):
        """Return one image attribution map for list and ndarray SHAP formats."""
        if self.shap_values is None:
            raise ValueError("Primero carga o calcula los valores SHAP.")

        if isinstance(self.shap_values, list):
            values = np.asarray(self.shap_values[class_idx])[instance_idx]
        else:
            values_all = np.asarray(self.shap_values)
            if values_all.ndim == 5:  # (N,H,W,C,K)
                values = values_all[instance_idx, ..., class_idx]
            elif values_all.ndim == 4:  # (N,H,W,K)
                values = values_all[instance_idx, ..., class_idx]
            else:
                raise ValueError(f"Formato SHAP no reconocido: {values_all.shape}")

        values = np.asarray(values)

        # Para imágenes, cada posición espacial constituye el píxel que se
        # visualizará. En imágenes RGB se suman las contribuciones de sus tres
        # canales para obtener un único valor diferencial por píxel. La suma,
        # a diferencia del promedio, conserva la aditividad de las atribuciones.
        if values.ndim == 3:
            values = np.sum(values, axis=-1)

        if values.ndim != 2:
            raise ValueError(
                "Se esperaba un mapa SHAP 2D después de agregar los canales; "
                f"se recibió {values.shape}."
            )

        return values

    @staticmethod
    def _class_name_to_index(class_name, class_names):
        inverse = {name: idx for idx, name in class_names.items()}
        if class_name not in inverse:
            raise ValueError(f"Clase desconocida: {class_name}")
        return inverse[class_name]

    @staticmethod
    def _finish_figure(fig, save_path=None, show=True, dpi=600):
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Figura guardada en: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_probability_differences(
        self,
        instance_idx,
        class_names=None,
        instance_label="Instance",
        top_k=10,
        save_path=None,
        show=True,
    ):
        probabilities = self.model.predict(
            self.x_eval[instance_idx : instance_idx + 1], verbose=0
        )[0]
        classes = [
            class_names[i] if class_names else f"Class {i}"
            for i in range(len(probabilities))
        ]
        differences = [
            (abs(probabilities[i] - probabilities[j]), classes[i], classes[j])
            for i in range(len(probabilities))
            for j in range(i + 1, len(probabilities))
        ]
        differences.sort(key=lambda item: item[0], reverse=True)
        plotted = differences if top_k is None else differences[:top_k]

        fig, axis = plt.subplots(figsize=(14, 8))
        labels = [f"{item[1]} vs {item[2]}" for item in plotted]
        sns.barplot(
            x=[item[0] for item in plotted],
            y=labels,
            hue=labels,
            palette="viridis",
            legend=False,
            ax=axis,
        )
        axis.set_title(
            f"Absolute Differences between Class Probabilities for {instance_label}",
            fontsize=34,
        )
        axis.set_xlabel(r"Absolute Differences $|\Delta P|$", fontsize=34)
        axis.set_ylabel(r"Class Pairs (descending $|\Delta P|$)", fontsize=34)
        axis.tick_params(axis="both", labelsize=28)
        fig.tight_layout()
        self._finish_figure(fig, save_path, show)
        return differences

    def explain_selected_pairs(
        self,
        instance_idx,
        selected_pairs,
        class_names,
        instance_label="Instance",
        top_n=10,
        output_dir=None,
        show=True,
    ):
        if not isinstance(class_names, dict):
            raise ValueError("class_names debe ser un diccionario {índice: nombre}.")

        saved_paths = []
        for class_i_name, class_j_name in selected_pairs:
            idx_i = self._class_name_to_index(class_i_name, class_names)
            idx_j = self._class_name_to_index(class_j_name, class_names)
            shap_difference = self._values_for_class(instance_idx, idx_i) - self._values_for_class(
                instance_idx, idx_j
            )

            coordinates = [
                f"px_{row}_{column}"
                for row in range(shap_difference.shape[0])
                for column in range(shap_difference.shape[1])
            ]
            frame = pd.DataFrame(
                {
                    "Pixel": coordinates,
                    "SHAP_Difference": shap_difference.reshape(-1),
                }
            )
            frame = frame.loc[
                frame["SHAP_Difference"].abs().sort_values(ascending=False).index
            ].head(top_n)

            fig, axis = plt.subplots(figsize=(12, 8))
            sns.barplot(
                x="SHAP_Difference",
                y="Pixel",
                data=frame,
                hue="Pixel",
                palette="viridis",
                legend=False,
                ax=axis,
            )
            axis.set_title(
                f"Top $|\\Delta S|$ between {class_i_name} and {class_j_name}\n{instance_label}",
                fontsize=16,
            )
            axis.set_xlabel(r"SHAP differences $\Delta S$", fontsize=13)
            axis.set_ylabel(r"Pixels (descending $|\Delta S|$)", fontsize=13)
            fig.tight_layout()

            save_path = None
            if output_dir is not None:
                safe_i = class_i_name.replace("/", "_").replace(" ", "_")
                safe_j = class_j_name.replace("/", "_").replace(" ", "_")
                save_path = Path(output_dir) / f"deltaS_{safe_i}_vs_{safe_j}.png"
                saved_paths.append(save_path)
            self._finish_figure(fig, save_path, show)
        return saved_paths

    def plot_shap_heatmap(
        self,
        instance_idx,
        class_i,
        class_j,
        class_names,
        instance_label="Instance",
        cmap="PiYG",
        alpha=0.6,
        show_original=True,
        save_path=None,
        show=True,
    ):
        if not isinstance(class_names, dict):
            raise ValueError("class_names debe ser un diccionario {índice: nombre}.")

        idx_i = self._class_name_to_index(class_i, class_names)
        idx_j = self._class_name_to_index(class_j, class_names)
        shap_difference = self._values_for_class(instance_idx, idx_i) - self._values_for_class(
            instance_idx, idx_j
        )
        original = self.x_eval[instance_idx].squeeze()
        max_abs = np.max(np.abs(shap_difference)) + 1e-12

        fig = plt.figure(
            figsize=(3.35, 2.5 if show_original else 2.2),
            dpi=600,
            constrained_layout=True,
        )
        if show_original:
            original_axis = fig.add_subplot(1, 2, 1)
            original_axis.imshow(original, cmap="gray", interpolation="nearest")
            original_axis.set_title("Original", fontsize=7, pad=2)
            original_axis.axis("off")
            heatmap_axis = fig.add_subplot(1, 2, 2)
        else:
            heatmap_axis = fig.add_subplot(1, 1, 1)

        heatmap_axis.imshow(original, cmap="gray", alpha=0.85, interpolation="nearest")
        image = heatmap_axis.imshow(
            shap_difference,
            cmap=cmap,
            alpha=alpha,
            vmin=-max_abs,
            vmax=max_abs,
            interpolation="nearest",
        )
        heatmap_axis.set_title(
            f"{class_i} vs {class_j}\n{instance_label}", fontsize=6, pad=2
        )
        heatmap_axis.axis("off")
        colorbar = fig.colorbar(image, ax=heatmap_axis, fraction=0.08, pad=0.02)
        colorbar.set_label(r"$\Delta S$", fontsize=7, labelpad=2)
        colorbar.ax.tick_params(labelsize=6, length=2)

        self._finish_figure(fig, save_path, show)
        return shap_difference
