# SHAPExplainer2
# ------------------
# Esta clase permite el uso de valores SHAP para interpretar modelos de clasificación multiclase, siendo compatible con modelos basados en árboles, lineales y modelos caja negra.
#
# Consideraciones importantes:
# - Todos los modelos utilizados deben implementar el metodo "predict_proba()", ya que es necesario para calcular las probabilidades de clase.
# - En el caso específico del modelo "SVC" de "scikit-learn", es obligatorio instanciarlo con "probability=True" para habilitar la funcionalidad de predicción de probabilidades.
# - La clase "SHAPExplainer2" selecciona automáticamente el tipo de explicador SHAP más apropiado:
#     - TreeExplainer para modelos basados en árboles,
#     - LinearExplainer para modelos lineales,
#     - KernelExplainer para modelos no lineales o caja negra.
# Esto permite utilizar una amplia variedad de modelos sin modificar el flujo principal del código, manteniendo así la flexibilidad y modularidad del sistema.

from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

class SHAPExplainer:
    def __init__(self, csv_path, model_class, target_column, path_save_images, n_splits=10, test_size=0.3, random_state=42, sep=','):
        self.csv_path = csv_path
        self.sep = sep
        self.model_class = model_class
        self.name_model = model_class.func.__name__ if isinstance(model_class, partial) else model_class.__name__
        self.database_name = Path(csv_path).stem
        self.target_column = target_column
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.modelos_guardados = []
        self.resultados = []
        self.shap_values = None
        self.modelo_entrenado_70 = None
        self.explainer = None
        self.exist_database_path = os.path.exists(self.csv_path)
        self._load_data()
        self.path_images = path_save_images
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images)

    def _load_data(self):
        if not self.exist_database_path:
            print('No existe el archivo, descargalo primero a la raiz del proyecto')
            return

        df = pd.read_csv(self.csv_path, sep=self.sep)
        self.df = df.copy()
        self.y = df[self.target_column].values
        df_features = df.drop(columns=[self.target_column])
        for col in df_features.select_dtypes(include='object').columns:
            df_features[col] = pd.Categorical(df_features[col]).codes
        self.X = df_features.values
        self.X_norm = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_norm, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state)

    def _init_explainer(self):
        model_name = self.modelo_entrenado_70.__class__.__name__
        if "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name:
            self.explainer = shap.TreeExplainer(self.modelo_entrenado_70)
        elif "LogisticRegression" in model_name or "Linear" in model_name:
            self.explainer = shap.LinearExplainer(self.modelo_entrenado_70, self.X_train)
        else:
            background = shap.sample(self.X_train, 100, random_state=self.random_state)
            self.explainer = shap.KernelExplainer(self.modelo_entrenado_70.predict_proba, background)

    def train_and_evaluate(self):
        if not self.exist_database_path:
            return

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (train_index, val_index) in enumerate(cv.split(self.X_train, self.y_train), start=1):
            X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
            y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]
            try:
                modelo = self.model_class(random_state=self.random_state)
            except TypeError:
                modelo = self.model_class()
            modelo.fit(X_fold_train, y_fold_train)
            self.modelos_guardados.append(modelo)
            accuracy = modelo.score(X_fold_val, y_fold_val)
            self.resultados.append({'Pliegue': f'Pliegue {fold}', 'Accuracy': accuracy})

        print("\nResultados de Validación Cruzada:")
        df = pd.DataFrame(self.resultados)
        df.index = df.index + 1
        print(df)

        try:
            self.modelo_entrenado_70 = self.model_class(random_state=self.random_state)
        except TypeError:
            self.modelo_entrenado_70 = self.model_class()
        self.modelo_entrenado_70.fit(self.X_train, self.y_train)
        self._init_explainer()
        print("\nModelo final entrenado sobre el 70% de los datos.")

    def calculate_shap_values(self):
        if not self.exist_database_path:
            return

        if self.modelo_entrenado_70 is None or self.explainer is None:
            raise ValueError("Debe entrenar el modelo y generar el explainer antes de calcular los valores SHAP.")
        self.shap_values = self.explainer.shap_values(self.X_test)
        print("Valores SHAP calculados con forma:", np.array(self.shap_values).shape)

    def save_shap_values(self, filename="shap_values.pickle"):
        if not self.exist_database_path:
            return

        if self.shap_values is None:
            raise ValueError("Debe calcular los valores SHAP antes de guardarlos.")
        with open(filename, 'wb') as handle:
            pickle.dump(self.shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Valores SHAP guardados en {filename}")

    def load_or_compute(self, filename="shap_values.pickle"):
        """Carga los valores SHAP desde disco si existen; si no, entrena el modelo, los calcula y los guarda."""
        if os.path.exists(filename):
            with open(filename, 'rb') as handle:
                self.shap_values = pickle.load(handle)
            # El modelo entrenado es necesario para predict_proba; entrenar igualmente pero sin recalcular SHAP
            self.train_and_evaluate()
            print(f"Valores SHAP cargados desde {filename}")
        else:
            self.train_and_evaluate()
            self.calculate_shap_values()
            self.save_shap_values(filename)
            print(f"Valores SHAP calculados y guardados en {filename}")

    def plot_probability_differences(self, instance, instance_type="test", class_names=None, instance_label="Instancia"):
        if not self.exist_database_path:
            return

        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de usar SHAP.")

        if instance_type == "test":
            probabilities = self.modelo_entrenado_70.predict_proba([self.X_test[instance]])[0]
        else:
            instance_scaled = self.scaler.transform([instance])
            probabilities = self.modelo_entrenado_70.predict_proba(instance_scaled)[0]

        num_classes = len(probabilities)
        classes = [class_names[i] if class_names else f"Clase {i}" for i in range(num_classes)]

        differences = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diff = abs(probabilities[i] - probabilities[j])
                differences.append((diff, classes[i], classes[j]))

        diffs = sorted(differences, key=lambda x: x[0], reverse=True)

        plt.figure(figsize=(16, 10))
        labels = [f"{diff[1]} vs {diff[2]}" for diff in diffs]
        sns.barplot(
            x=[diff[0] for diff in diffs],
            y=labels,
            hue=labels,
            palette="viridis",
            legend=False
        )
        plt.title(f"Diferencias Absolutas entre Probabilidades - {instance_label}", fontsize=22)
        plt.xlabel(r"Diferencias Absolutas $|\mathrm{\Delta P}|$", fontsize=20)
        plt.ylabel("Pares de Clases", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.path_images}/Grafica_probability_diff_values_{instance_label}.png", dpi=300, bbox_inches='tight')
        plt.close()

        return diffs

    def explain_selected_pairs(self, instance, selected_pairs, instance_type="test", class_names=None, instance_label="Instancia"):
        if not self.exist_database_path:
            return

        if self.modelo_entrenado_70 is None or self.explainer is None:
            raise ValueError("Debe entrenar el modelo y generar el explainer antes de usar SHAP.")

        if instance_type == "test":
            if isinstance(self.shap_values, list):
                shap_values_instance = [shap_class[instance] for shap_class in self.shap_values]
            else:
                shap_values_instance = [self.shap_values[instance, :, i] for i in range(self.shap_values.shape[2])]
            probabilities_instance = self.modelo_entrenado_70.predict_proba([self.X_test[instance]])[0]
        else:
            instance_scaled = self.scaler.transform([instance])
            shap_values_raw = self.explainer.shap_values(instance_scaled)
            probabilities_instance = self.modelo_entrenado_70.predict_proba(instance_scaled)[0]
            if isinstance(shap_values_raw, list):
                shap_values_instance = [shap_values_raw[i][0] for i in range(len(shap_values_raw))]
            else:
                shap_values_instance = [shap_values_raw[0, :, i] for i in range(shap_values_raw.shape[2])]

        inverse_class_names = {v: k for k, v in class_names.items()} if class_names else {f"Clase {i}": i for i in range(len(probabilities_instance))}

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
                f"Diferencias en Valores SHAP\nEntre Clases '{class_i}' y '{class_j}' - {instance_label}",
                fontsize=26
            )
            plt.xlabel(r"Diferencias SHAP $\mathrm{\Delta S}$", fontsize=24)
            plt.ylabel("Características en orden descendente", fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(f"{self.path_images}/Grafica_diff_values_shap_between_class_{class_i}__{class_j}-{instance_label}.png", dpi=300, bbox_inches='tight')
            plt.close()
