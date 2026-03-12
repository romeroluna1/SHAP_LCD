import numpy as np
import pandas as pd
import shap
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, matthews_corrcoef
import os


class SHAPExplainer:
    def __init__(
        self,
        csv_path,
        model_class,
        target_column,
        n_splits=10,
        test_size=0.3,
        random_state=42,
        target_as_str=False,
    ):
        self.csv_path = csv_path
        self.model_class = model_class
        self.target_column = target_column
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.target_as_str = target_as_str

        self.scaler = MinMaxScaler()
        self.modelos_guardados = []
        self.resultados = []
        self.shap_values = None
        self.modelo_entrenado_70 = None
        self.explainer = None

        self._load_data()

    # -------------------------
    # DATA
    # -------------------------
    def _load_data(self):
        df = pd.read_csv(self.csv_path)

        if self.target_as_str:
            df[self.target_column] = df[self.target_column].astype(str)

        self.df = df.copy()
        self.y = df[self.target_column].values
        self.X = df.drop(columns=[self.target_column]).values

        print("\nDistribución de clases (dataset completo):")
        print(pd.Series(self.y).value_counts().sort_index())

        self.X_norm = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_norm,
            self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state
        )

    # -------------------------
    # EXPLAINER INIT
    # -------------------------
    def _init_explainer(self):
        model_name = self.modelo_entrenado_70.__class__.__name__
        if ("Tree" in model_name) or ("Forest" in model_name) or ("Boosting" in model_name) or ("XGB" in model_name):
            self.explainer = shap.TreeExplainer(self.modelo_entrenado_70)
        elif ("LogisticRegression" in model_name) or ("Linear" in model_name):
            self.explainer = shap.LinearExplainer(self.modelo_entrenado_70, self.X_train)
        else:
            background = shap.sample(self.X_train, 100, random_state=self.random_state)
            self.explainer = shap.KernelExplainer(self.modelo_entrenado_70.predict_proba, background)

    # -------------------------
    # TRAIN / CV
    # -------------------------
    def train_and_evaluate(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.resultados = []
        self.modelos_guardados = []

        for fold, (train_index, val_index) in enumerate(cv.split(self.X_train, self.y_train), start=1):
            X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
            y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]

            try:
                modelo = self.model_class(random_state=self.random_state)
            except TypeError:
                modelo = self.model_class()

            modelo.fit(X_fold_train, y_fold_train)
            self.modelos_guardados.append(modelo)

            y_pred = modelo.predict(X_fold_val)

            f1_macro = f1_score(y_fold_val, y_pred, average="macro")
            mcc = matthews_corrcoef(y_fold_val, y_pred)
            acc = modelo.score(X_fold_val, y_fold_val)

            self.resultados.append({
                "Pliegue": f"Pliegue {fold}",
                "Accuracy": acc,
                "F1_macro": f1_macro,
                "MCC": mcc
            })

        df_res = pd.DataFrame(self.resultados)
        print("\nResultados de Validación Cruzada (por pliegue):")
        print(df_res.to_string(index=False))

        resumen = df_res[["Accuracy", "F1_macro", "MCC"]].agg(["mean", "std"]).T
        resumen["mean±std"] = (
            resumen["mean"].map(lambda x: f"{x:.4f}") + " ± " + resumen["std"].map(lambda x: f"{x:.4f}")
        )

        print("\nResumen (media ± desviación estándar):")
        print(resumen[["mean±std"]].to_string())

        try:
            self.modelo_entrenado_70 = self.model_class(random_state=self.random_state)
        except TypeError:
            self.modelo_entrenado_70 = self.model_class()

        self.modelo_entrenado_70.fit(self.X_train, self.y_train)
        self._init_explainer()
        print("\nModelo final entrenado sobre el 70% de los datos.")

    # -------------------------
    # SHAP: CALC / SAVE / LOAD / POLICY (auto|always|never)
    # -------------------------
    def calculate_shap_values(self):
        if self.modelo_entrenado_70 is None or self.explainer is None:
            raise ValueError("Debe entrenar el modelo y generar el explainer antes de calcular los valores SHAP.")
        self.shap_values = self.explainer.shap_values(self.X_test)
        print("Valores SHAP calculados con forma:", np.array(self.shap_values).shape)

    def save_shap_values(self, filename="shap_values.pickle"):
        if self.shap_values is None:
            raise ValueError("Debe calcular los valores SHAP antes de guardarlos.")
        with open(filename, "wb") as handle:
            pickle.dump(self.shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Valores SHAP guardados en {filename}")

    def load_shap_values(self, filename="shap_values.pickle"):
        with open(filename, "rb") as handle:
            self.shap_values = pickle.load(handle)
        print(f"Valores SHAP cargados desde {filename} con forma:", np.array(self.shap_values).shape)

    def ensure_shap_values(self, filename="shap_values.pickle", mode="auto"):
        """
        mode:
          - "auto": si existe -> cargar; si no -> calcular y guardar
          - "always": recalcular y guardar
          - "never": solo cargar (si no existe -> error)
        """
        mode = str(mode).lower().strip()
        exists = os.path.exists(filename)

        if mode == "always":
            self.calculate_shap_values()
            self.save_shap_values(filename)
            return

        if mode == "never":
            if not exists:
                raise FileNotFoundError(
                    f"No existe {filename}. mode='never' exige SHAP precalculado."
                )
            self.load_shap_values(filename)
            return

        if mode == "auto":
            if exists:
                self.load_shap_values(filename)
            else:
                self.calculate_shap_values()
                self.save_shap_values(filename)
            return

        raise ValueError("mode debe ser 'auto', 'always' o 'never'.")

    # -------------------------
    # PLOTS / ALGORITMOS (sin cambios funcionales)
    # -------------------------
    def plot_probability_differences(self, instance, instance_type="test", class_names=None, instance_label="Instancia"):
        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de usar SHAP.")

        if instance_type == "test":
            probabilities = self.modelo_entrenado_70.predict_proba([self.X_test[instance]])[0]
        else:
            instance_scaled = self.scaler.transform([instance])
            probabilities = self.modelo_entrenado_70.predict_proba(instance_scaled)[0]

        num_classes = len(probabilities)
        classes = [class_names[i] if class_names else f"Clase {i}" for i in range(num_classes)]

        def _wrap_if_long(name, max_len=20):
            name = str(name)
            if len(name) <= max_len:
                return name
            mid = len(name) // 2
            left_space = name.rfind(" ", 0, mid)
            right_space = name.find(" ", mid)
            if left_space == -1 and right_space == -1:
                return name[:max_len] + "\n" + name[max_len:]
            cut = left_space if (left_space != -1 and (right_space == -1 or (mid - left_space) <= (right_space - mid))) else right_space
            return name[:cut].strip() + "\n" + name[cut:].strip()

        do_wrap = any(len(str(c)) > 10 for c in classes)

        differences = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diff = abs(probabilities[i] - probabilities[j])
                differences.append((diff, classes[i], classes[j]))

        diffs = sorted(differences, key=lambda x: x[0], reverse=True)

        plt.figure(figsize=(16, 10))

        if do_wrap:
            labels = [f"{_wrap_if_long(d[1], 20)}\nvs\n{_wrap_if_long(d[2], 20)}" for d in diffs]
        else:
            labels = [f"{d[1]} vs {d[2]}" for d in diffs]

        ax = sns.barplot(
            x=[d[0] for d in diffs],
            y=labels,
            hue=labels,
            palette="viridis",
            legend=False
        )

        for t in ax.get_yticklabels():
            t.set_multialignment("center")
            t.set_linespacing(0.85)
            t.set_ha("right")

        plt.title(f"Absolute Differences Between Class Probabilities \n for an {instance_label}", fontsize=34)
        plt.xlabel(r"Absolute Differences $|\mathrm{\Delta P}|$", fontsize=34)
        plt.ylabel("Class Pairs", fontsize=34)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.show()

        return diffs

    def explain_selected_pairs(self, instance, selected_pairs, instance_type="test", class_names=None, instance_label="Instancia"):
        if self.modelo_entrenado_70 is None or self.explainer is None:
            raise ValueError("Debe entrenar el modelo y generar el explainer antes de usar SHAP.")

        if instance_type == "test":
            if self.shap_values is None:
                raise ValueError("self.shap_values es None. Usa ensure_shap_values(...) o calculate_shap_values().")

            if isinstance(self.shap_values, list):
                shap_values_instance = [shap_class[instance] for shap_class in self.shap_values]
            else:
                shap_values_instance = [self.shap_values[instance, :, i] for i in range(self.shap_values.shape[2])]
        else:
            instance_scaled = self.scaler.transform([instance])
            shap_values_raw = self.explainer.shap_values(instance_scaled)
            if isinstance(shap_values_raw, list):
                shap_values_instance = [shap_values_raw[i][0] for i in range(len(shap_values_raw))]
            else:
                shap_values_instance = [shap_values_raw[0, :, i] for i in range(shap_values_raw.shape[2])]

        if class_names:
            inverse_class_names = {v: k for k, v in class_names.items()}
        else:
            inverse_class_names = {f"Clase {i}": i for i in range(len(shap_values_instance))}

        for class_i, class_j in selected_pairs:
            idx_i = inverse_class_names[class_i]
            idx_j = inverse_class_names[class_j]

            shap_diff = shap_values_instance[idx_i] - shap_values_instance[idx_j]

            shap_diff_df = pd.DataFrame({
                "Feature": self.df.drop(columns=[self.target_column]).columns,
                "SHAP_Difference": shap_diff
            }).sort_values(by="SHAP_Difference", key=abs, ascending=False)

            plt.figure(figsize=(18, 14))
            sns.barplot(
                x="SHAP_Difference",
                y="Feature",
                hue="Feature",
                data=shap_diff_df.head(10),
                palette="viridis",
                legend=False
            )
            plt.title(
                f"Differences in SHAP Values of Main Features\nBetween '{class_i}' and '{class_j}' for {instance_label}",
                fontsize=30
            )
            plt.xlabel(r"SHAP Differences $\mathrm{\Delta S}$", fontsize=30)
            plt.ylabel("Features in descending order", fontsize=30)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.tight_layout()
            plt.show()

    # -------------------------
    # VIOLINS / BARS (igual que tu versión)
    # -------------------------
    def plot_violin_distributions(
        self,
        class_names=None,
        max_features=10,
        sort_by="variance",
        use_normalized=True,
        save_dir=None,
        dpi=600,
        figsize=(12, 6),
        show=True,
        show_legend=False,
        violin_alpha=0.6,
        cmap_min=0.2,
        cmap_max=0.8,
        overlay_box=True,
        box_width=0.20,
        whis=1.5,
        overlay_color="black",
        fliersize=8,
        flier_alpha=0.97
    ):
        X_plot = self.X_norm if use_normalized else self.X
        feature_names = self.df.drop(columns=[self.target_column]).columns.tolist()
        df_plot = pd.DataFrame(X_plot, columns=feature_names)

        y_series = pd.Series(self.y)

        if class_names is not None:
            def _to_int_safe(v):
                try:
                    return int(v)
                except Exception:
                    return None

            y_int = y_series.map(_to_int_safe)
            if y_int.notna().all():
                df_plot["Class"] = y_int.map(lambda k: class_names.get(k, str(k))).astype(str)
            else:
                df_plot["Class"] = y_series.astype(str)
        else:
            df_plot["Class"] = y_series.astype(str)

        if class_names is not None:
            ordered_labels = [str(class_names[k]) for k in sorted(class_names.keys())]
            ordered_labels = [c for c in ordered_labels if c in df_plot["Class"].unique()]
        else:
            ordered_labels = sorted(df_plot["Class"].unique())

        df_plot["Class"] = pd.Categorical(df_plot["Class"], categories=ordered_labels, ordered=True)

        selected_features = feature_names
        if sort_by == "variance":
            variances = df_plot[feature_names].var().sort_values(ascending=False)
            selected_features = variances.index.tolist()
        selected_features = selected_features[:max_features]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        colors = sns.color_palette("viridis", as_cmap=True)
        positions = np.linspace(cmap_min, cmap_max, len(ordered_labels))
        pal = [colors(p) for p in positions]

        for feat in selected_features:
            plt.figure(figsize=figsize)

            ax = sns.violinplot(
                data=df_plot,
                x="Class",
                y=feat,
                hue="Class",
                palette=pal,
                inner=None,
                cut=0,
                legend=False
            )

            for coll in ax.collections:
                try:
                    coll.set_alpha(violin_alpha)
                    coll.set_edgecolor(None)
                    coll.set_linewidth(0)
                except Exception:
                    pass

            for ln in ax.lines:
                ln.set_color(overlay_color)
                ln.set_linewidth(1.2)
                ln.set_alpha(0.9)

            if overlay_box:
                sns.boxplot(
                    data=df_plot,
                    x="Class",
                    y=feat,
                    width=box_width,
                    showfliers=True,
                    whis=whis,
                    boxprops={"facecolor": "none", "edgecolor": overlay_color, "linewidth": 1.2},
                    whiskerprops={"color": overlay_color, "linewidth": 1.2},
                    capprops={"color": overlay_color, "linewidth": 1.2},
                    medianprops={"color": overlay_color, "linewidth": 1.2},
                    flierprops={
                        "marker": "o",
                        "markersize": fliersize,
                        "markerfacecolor": overlay_color,
                        "markeredgecolor": overlay_color,
                        "alpha": flier_alpha
                    }
                )

            title_feat = feat.replace("(cm)", "").strip() if use_normalized else feat
            ax.set_title(f"Class-wise distribution (violin plot) - {title_feat}", fontsize=28)
            ax.set_xlabel("Class", fontsize=28)
            ax.set_ylabel("Value (normalized)" if use_normalized else "Value", fontsize=28)

            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.tight_layout()

            if show_legend:
                handles = [mpatches.Patch(color=pal[i], label=ordered_labels[i]) for i in range(len(ordered_labels))]
                ax.legend(handles=handles, title="Class", loc="upper right", frameon=True)

            if save_dir is not None:
                out_path = os.path.join(save_dir, f"violin_{feat}.png")
                plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

    def plot_global_violin_by_class(
        self,
        class_names=None,
        use_normalized=True,
        mode="pooled_features",
        dataset_name=None,
        save_path=None,
        dpi=600,
        figsize=(8, 5),
        show=True,
        show_legend=False,
        violin_alpha=0.75,
        cmap_min=0.2,
        cmap_max=0.8,
        overlay_box=True,
        box_width=0.3,
        whis=1.5,
        overlay_color="black",
        fliersize=8,
        flier_alpha=0.97
    ):
        X_plot = self.X_norm if use_normalized else self.X
        y_series = pd.Series(self.y)

        if class_names is not None:
            def _to_int_safe(v):
                try:
                    return int(v)
                except Exception:
                    return None

            y_int = y_series.map(_to_int_safe)
            if y_int.notna().all():
                class_labels = y_int.map(lambda k: class_names.get(k, str(k))).astype(str)
                ordered_labels = [str(class_names[k]) for k in sorted(class_names.keys())]
                ordered_labels = [c for c in ordered_labels if c in set(class_labels)]
            else:
                class_labels = y_series.astype(str)
                ordered_labels = sorted(class_labels.unique())
        else:
            class_labels = y_series.astype(str)
            ordered_labels = sorted(class_labels.unique())

        if mode == "pooled_features":
            n_samples, n_feats = X_plot.shape
            values = X_plot.reshape(-1)
            classes_rep = np.repeat(class_labels.values, n_feats)
            df_long = pd.DataFrame({"Class": classes_rep, "Value": values})
            y_label = "Value (normalized)" if use_normalized else "Value"
            title_base = "Global class-wise distribution (pooled features)"
        elif mode == "instance_mean":
            inst_mean = X_plot.mean(axis=1)
            df_long = pd.DataFrame({"Class": class_labels.values, "Value": inst_mean})
            y_label = "Mean value (normalized)" if use_normalized else "Mean value"
            title_base = "Global class-wise distribution (instance mean)"
        else:
            raise ValueError("mode debe ser 'pooled_features' o 'instance_mean'")

        title = f"{dataset_name} — {title_base}" if dataset_name else title_base

        df_long["Class"] = pd.Categorical(df_long["Class"], categories=ordered_labels, ordered=True)

        colors = sns.color_palette("viridis", as_cmap=True)
        positions = np.linspace(cmap_min, cmap_max, len(ordered_labels))
        pal = [colors(p) for p in positions]

        plt.figure(figsize=figsize)

        ax = sns.violinplot(
            data=df_long,
            x="Class",
            y="Value",
            hue="Class",
            palette=pal,
            inner=None,
            cut=0,
            legend=False
        )

        for coll in ax.collections:
            try:
                coll.set_alpha(violin_alpha)
                coll.set_edgecolor(None)
                coll.set_linewidth(0)
            except Exception:
                pass

        if overlay_box:
            sns.boxplot(
                data=df_long,
                x="Class",
                y="Value",
                width=box_width,
                showfliers=True,
                whis=whis,
                boxprops={"facecolor": "none", "edgecolor": overlay_color, "linewidth": 1.2},
                whiskerprops={"color": overlay_color, "linewidth": 1.2},
                capprops={"color": overlay_color, "linewidth": 1.2},
                medianprops={"color": overlay_color, "linewidth": 1.2},
                flierprops={
                    "marker": "o",
                    "markersize": fliersize,
                    "markerfacecolor": overlay_color,
                    "markeredgecolor": overlay_color,
                    "alpha": flier_alpha
                }
            )

        ax.set_title(title, fontsize=36)
        ax.set_xlabel("Class", fontsize=36)
        ax.set_ylabel(y_label, fontsize=36)
        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)
        plt.tight_layout()

        if show_legend:
            handles = [mpatches.Patch(color=pal[i], label=ordered_labels[i]) for i in range(len(ordered_labels))]
            ax.legend(handles=handles, title="Class", loc="upper right", frameon=True)

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_class_distribution_bar(
        self,
        class_names=None,
        dataset_name=None,
        save_path=None,
        dpi=600,
        figsize=(8, 5),
        show=True,
        cmap_min=0.2,
        cmap_max=0.8,
        bar_alpha=0.95,
        show_values=True
    ):
        y_series = pd.Series(self.y)

        if class_names is not None:
            def _to_int_safe(v):
                try:
                    return int(v)
                except Exception:
                    return None

            y_int = y_series.map(_to_int_safe)
            if y_int.notna().all():
                class_labels = y_int.map(lambda k: class_names.get(k, str(k))).astype(str)
                ordered_labels = [str(class_names[k]) for k in sorted(class_names.keys())]
                ordered_labels = [c for c in ordered_labels if c in set(class_labels)]
            else:
                class_labels = y_series.astype(str)
                ordered_labels = sorted(class_labels.unique())
        else:
            class_labels = y_series.astype(str)
            ordered_labels = sorted(class_labels.unique())

        counts = class_labels.value_counts().reindex(ordered_labels)
        perc = (counts / counts.sum()) * 100

        df_bar = pd.DataFrame({"Class": ordered_labels, "Percent": perc.values})

        colors = sns.color_palette("viridis", as_cmap=True)
        positions = np.linspace(cmap_min, cmap_max, len(ordered_labels))
        pal = [colors(p) for p in positions]

        plt.figure(figsize=figsize)

        ax = sns.barplot(
            data=df_bar,
            x="Class",
            y="Percent",
            hue="Class",
            palette=pal,
            edgecolor=None,
            legend=False
        )
        ymax = df_bar["Percent"].max()
        ax.set_ylim(0, ymax * 1.10)

        for patch in ax.patches:
            patch.set_alpha(bar_alpha)

        title_base = "Class distribution"
        title = f"{title_base}-{dataset_name}" if dataset_name else title_base

        ax.set_title(title, fontsize=36)
        ax.set_xlabel("Class", fontsize=36)
        ax.set_ylabel("Percentage (%)", fontsize=36)
        plt.xticks(fontsize=34, rotation=0)
        plt.yticks(fontsize=34)
        plt.tight_layout()

        if show_values:
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    (p.get_x() + p.get_width() / 2., h),
                    ha="center",
                    va="bottom",
                    fontsize=28,
                    xytext=(0, 4),
                    textcoords="offset points"
                )

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    # -------------------------
    # SHAP_LCD: helpers
    # -------------------------
    @staticmethod
    def _cosine_sim(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

    def _get_pair_indices(self, pair, class_names=None):
        """
        Devuelve:
          (ci_idx, cj_idx, ci_lbl, cj_lbl)
        donde:
          - ci_idx/cj_idx: índices 0..K-1 (para probs/SHAP)
          - ci_lbl/cj_lbl: labels reales tal como están en modelo.classes_ (para filtrar y_test)
        Compatible con:
          - datasets con y numérica (Iris/Wine)
          - datasets con y string (Vehicle con target_as_str=True)
          - pair como índices, o como labels, o como nombres (si se pasa class_names)
        """
        if self.modelo_entrenado_70 is None:
            raise ValueError("El modelo debe estar entrenado antes de resolver el par de clases.")

        classes = list(self.modelo_entrenado_70.classes_)  # labels reales (pueden ser str o int)

        def _resolve_one(x):
            # 1) si viene "nombre" (bus/saab) -> convertir a índice usando class_names
            if class_names is not None:
                inv_names = {str(v): k for k, v in class_names.items()}  # nombre -> key
                if str(x) in inv_names:
                    x = inv_names[str(x)]  # ahora x es key (normalmente int)

            # 2) si x coincide con label exacto en classes
            if x in classes:
                return classes.index(x), x

            # 3) si str(x) coincide con label (caso Vehicle: x=0, label="0")
            sx = str(x)
            if sx in [str(c) for c in classes]:
                # encontrar el match exacto en classes por string
                for c in classes:
                    if str(c) == sx:
                        return classes.index(c), c

            # 4) si x es int y puede ser índice directo (fallback)
            if isinstance(x, (int, np.integer)) and 0 <= int(x) < len(classes):
                idx = int(x)
                return idx, classes[idx]

            raise ValueError(
                f"No pude resolver la clase '{x}'. classes_ del modelo: {classes}. "
                f"Si estás usando nombres, pasa class_names. Si y está como string, usa '0','2' o deja (0,2) (ya lo manejo)."
            )

        ci_idx, ci_lbl = _resolve_one(pair[0])
        cj_idx, cj_lbl = _resolve_one(pair[1])
        return ci_idx, cj_idx, ci_lbl, cj_lbl

    def _shap_lcd_vector_from_precomputed(self, idx_test, class_i, class_j, top_k=None, normalize="l2"):
        if self.shap_values is None:
            raise ValueError("self.shap_values es None. Usa ensure_shap_values(...) o calculate_shap_values().")

        if isinstance(self.shap_values, list):
            phi_i = np.array(self.shap_values[class_i][idx_test], dtype=float)
            phi_j = np.array(self.shap_values[class_j][idx_test], dtype=float)
        else:
            phi_i = np.array(self.shap_values[idx_test, :, class_i], dtype=float)
            phi_j = np.array(self.shap_values[idx_test, :, class_j], dtype=float)

        v = phi_i - phi_j

        if top_k is not None and top_k < v.size:
            idx = np.argsort(np.abs(v))[::-1][:top_k]
            v_sparse = np.zeros_like(v)
            v_sparse[idx] = v[idx]
            v = v_sparse

        if normalize == "l1":
            v = v / (np.sum(np.abs(v)) + 1e-12)
        elif normalize == "l2":
            v = v / (np.linalg.norm(v) + 1e-12)

        return v

    def shap_lcd_vector(self, x_scaled, class_i, class_j, top_k=None, normalize="l2"):
        shap_raw = self.explainer.shap_values(x_scaled.reshape(1, -1))

        if isinstance(shap_raw, list):
            phi_i = np.array(shap_raw[class_i][0], dtype=float)
            phi_j = np.array(shap_raw[class_j][0], dtype=float)
        else:
            phi_i = np.array(shap_raw[0, :, class_i], dtype=float)
            phi_j = np.array(shap_raw[0, :, class_j], dtype=float)

        v = phi_i - phi_j

        if top_k is not None and top_k < v.size:
            idx = np.argsort(np.abs(v))[::-1][:top_k]
            v_sparse = np.zeros_like(v)
            v_sparse[idx] = v[idx]
            v = v_sparse

        if normalize == "l1":
            v = v / (np.sum(np.abs(v)) + 1e-12)
        elif normalize == "l2":
            v = v / (np.linalg.norm(v) + 1e-12)

        return v

    # -------------------------
    # METRICS
    # -------------------------
    def stability_shap_lcd(self, x_scaled, class_i, class_j, M=50, sigma=0.02, top_k=10, normalize="l2",
                          seed=42, base_exp=None):
        rng = np.random.default_rng(seed)
        if base_exp is None:
            base_exp = self.shap_lcd_vector(x_scaled, class_i, class_j, top_k=top_k, normalize=normalize)

        vals = []
        for _ in range(M):
            x_p = x_scaled + rng.normal(0.0, sigma, size=x_scaled.shape)
            x_p = np.clip(x_p, 0.0, 1.0)
            v_p = self.shap_lcd_vector(x_p, class_i, class_j, top_k=top_k, normalize=normalize)
            vals.append(np.linalg.norm(v_p - base_exp) ** 2)

        return float(np.mean(vals))

    def consistency_shap_lcd(self, x_scaled, class_i, class_j, M=50, sigma=0.02, top_k=10,
                             tau=0.90, normalize="l2", seed=42, base_pred=None, base_exp=None):
        rng = np.random.default_rng(seed)

        if base_pred is None:
            base_pred = self.modelo_entrenado_70.predict(x_scaled.reshape(1, -1))[0]
        if base_exp is None:
            base_exp = self.shap_lcd_vector(x_scaled, class_i, class_j, top_k=top_k, normalize=normalize)

        ok, cnt = 0, 0
        for _ in range(M):
            x_p = x_scaled + rng.normal(0.0, sigma, size=x_scaled.shape)
            x_p = np.clip(x_p, 0.0, 1.0)

            pred_p = self.modelo_entrenado_70.predict(x_p.reshape(1, -1))[0]
            if pred_p == base_pred:
                exp_p = self.shap_lcd_vector(x_p, class_i, class_j, top_k=top_k, normalize=normalize)
                sim = self._cosine_sim(base_exp, exp_p)
                ok += int(sim >= tau)
                cnt += 1

        return float(ok / cnt) if cnt > 0 else np.nan

    def consistency_implication_shap_lcd(
        self,
        x_scaled,
        class_i,
        class_j,
        M=50,
        sigma=0.02,
        top_k=10,
        tau=0.90,
        normalize="l2",
        seed=42,
        base_pred=None,
        base_exp=None
    ):
        rng = np.random.default_rng(seed)

        if base_pred is None:
            base_pred = self.modelo_entrenado_70.predict(x_scaled.reshape(1, -1))[0]
        if base_exp is None:
            base_exp = self.shap_lcd_vector(x_scaled, class_i, class_j, top_k=top_k, normalize=normalize)

        ok = 0
        antecedent_true = 0
        pred_keep = 0

        for _ in range(M):
            x_p = x_scaled + rng.normal(0.0, sigma, size=x_scaled.shape)
            x_p = np.clip(x_p, 0.0, 1.0)

            pred_p = self.modelo_entrenado_70.predict(x_p.reshape(1, -1))[0]
            if pred_p == base_pred:
                pred_keep += 1

            exp_p = self.shap_lcd_vector(x_p, class_i, class_j, top_k=top_k, normalize=normalize)
            sim = self._cosine_sim(base_exp, exp_p)

            A = (sim >= tau)
            B = (pred_p == base_pred)

            if A:
                antecedent_true += 1

            if (not A) or B:
                ok += 1

        C_imp = ok / M
        coverage = antecedent_true / M
        pred_keep_rate = pred_keep / M

        return float(C_imp), float(coverage), float(pred_keep_rate)

    # -------------------------
    # EVALUATION (par fijo): robusto para y string/int
    # -------------------------
    def evaluate_shap_lcd_pair(
        self,
        pair=(1, 2),
        class_names=None,
        filter_mode="true",
        n_instances=None,
        M=40,
        sigma=0.02,
        top_k=10,
        tau=0.90,
        normalize="l2",
        seed=42
    ):
        ci_idx, cj_idx, ci_lbl, cj_lbl = self._get_pair_indices(pair, class_names=class_names)

        # Filtrado usando labels reales (ci_lbl/cj_lbl)
        if filter_mode == "true":
            idxs = np.where(np.isin(self.y_test, [ci_lbl, cj_lbl]))[0]
        elif filter_mode == "pred":
            yhat = self.modelo_entrenado_70.predict(self.X_test)
            idxs = np.where(np.isin(yhat, [ci_lbl, cj_lbl]))[0]
        else:
            raise ValueError("filter_mode debe ser 'true' o 'pred'.")

        if len(idxs) == 0:
            raise ValueError(
                f"No hay instancias para evaluar con pair={pair} -> labels ({ci_lbl},{cj_lbl}). "
                f"Revisa class_names / codificación del target. Ejemplo Vehicle con target_as_str=True: "
                f"pair puede ser (0,2) o ('0','2'), ambos ya deberían funcionar."
            )

        if n_instances is not None:
            idxs = idxs[:min(int(n_instances), len(idxs))]

        # Etiquetas bonitas para imprimir
        li = str(class_names[int(str(ci_lbl))]) if class_names is not None else str(ci_lbl)
        lj = str(class_names[int(str(cj_lbl))]) if class_names is not None else str(cj_lbl)

        rows = []
        for t, idx in enumerate(idxs):
            x = self.X_test[idx]

            base_pred = self.modelo_entrenado_70.predict(x.reshape(1, -1))[0]

            base_exp = None
            if self.shap_values is not None:
                base_exp = self._shap_lcd_vector_from_precomputed(idx, ci_idx, cj_idx, top_k=top_k, normalize=normalize)

            S = self.stability_shap_lcd(
                x, ci_idx, cj_idx, M=M, sigma=sigma, top_k=top_k, normalize=normalize,
                seed=seed + t, base_exp=base_exp
            )

            C = self.consistency_shap_lcd(
                x, ci_idx, cj_idx, M=M, sigma=sigma, top_k=top_k, tau=tau, normalize=normalize,
                seed=seed + t, base_pred=base_pred, base_exp=base_exp
            )

            p = self.modelo_entrenado_70.predict_proba(x.reshape(1, -1))[0]
            dP = float(abs(p[ci_idx] - p[cj_idx]))

            rows.append({
                "idx_test": int(idx),
                "pair": f"{li} vs {lj}",
                "abs_deltaP_ij": dP,
                "Stability_S": float(S),
                "Consistency_C": float(C),
            })

        df_inst = pd.DataFrame(rows)

        summary = df_inst[["Stability_S", "Consistency_C"]].agg(["mean", "std"]).T
        summary["mean±std"] = (
            summary["mean"].map(lambda x: f"{x:.6f}") + " ± " + summary["std"].map(lambda x: f"{x:.6f}")
        )

        return df_inst, summary

    def evaluate_shap_lcd_pair_compare_consistency(
        self,
        pair=(1, 2),
        class_names=None,
        filter_mode="true",
        n_instances=None,
        M=40,
        sigma=0.02,
        top_k=10,
        tau=0.90,
        normalize="l2",
        seed=42
    ):
        ci_idx, cj_idx, ci_lbl, cj_lbl = self._get_pair_indices(pair, class_names=class_names)

        if filter_mode == "true":
            idxs = np.where(np.isin(self.y_test, [ci_lbl, cj_lbl]))[0]
        elif filter_mode == "pred":
            yhat = self.modelo_entrenado_70.predict(self.X_test)
            idxs = np.where(np.isin(yhat, [ci_lbl, cj_lbl]))[0]
        else:
            raise ValueError("filter_mode debe ser 'true' o 'pred'.")

        if len(idxs) == 0:
            raise ValueError(
                f"No hay instancias para evaluar con pair={pair} -> labels ({ci_lbl},{cj_lbl})."
            )

        if n_instances is not None:
            idxs = idxs[:min(int(n_instances), len(idxs))]

        li = str(class_names[int(str(ci_lbl))]) if class_names is not None else str(ci_lbl)
        lj = str(class_names[int(str(cj_lbl))]) if class_names is not None else str(cj_lbl)

        rows = []
        for t, idx in enumerate(idxs):
            x = self.X_test[idx]

            base_pred = self.modelo_entrenado_70.predict(x.reshape(1, -1))[0]

            base_exp = None
            if self.shap_values is not None:
                base_exp = self._shap_lcd_vector_from_precomputed(idx, ci_idx, cj_idx, top_k=top_k, normalize=normalize)

            S = self.stability_shap_lcd(
                x, ci_idx, cj_idx, M=M, sigma=sigma, top_k=top_k, normalize=normalize,
                seed=seed + t, base_exp=base_exp
            )

            C_predexp = self.consistency_shap_lcd(
                x, ci_idx, cj_idx, M=M, sigma=sigma, top_k=top_k, tau=tau, normalize=normalize,
                seed=seed + t, base_pred=base_pred, base_exp=base_exp
            )

            C_imp, coverage, pred_keep = self.consistency_implication_shap_lcd(
                x, ci_idx, cj_idx, M=M, sigma=sigma, top_k=top_k, tau=tau, normalize=normalize,
                seed=seed + t, base_pred=base_pred, base_exp=base_exp
            )

            p = self.modelo_entrenado_70.predict_proba(x.reshape(1, -1))[0]
            dP = float(abs(p[ci_idx] - p[cj_idx]))

            rows.append({
                "idx_test": int(idx),
                "pair": f"{li} vs {lj}",
                "abs_deltaP_ij": dP,
                "Stability_S": float(S),
                "Consistency_C_predexp": float(C_predexp),
                "Consistency_C_imp": float(C_imp),
                "Coverage_imp": float(coverage),
                "PredKeep_imp": float(pred_keep),
            })

        df_inst = pd.DataFrame(rows)

        cols = ["Stability_S", "Consistency_C_predexp", "Consistency_C_imp", "Coverage_imp", "PredKeep_imp"]
        summary = df_inst[cols].agg(["mean", "std"]).T
        summary["mean±std"] = (
            summary["mean"].map(lambda x: f"{x:.6f}") + " ± " + summary["std"].map(lambda x: f"{x:.6f}")
        )

        return df_inst, summary
