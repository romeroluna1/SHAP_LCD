# Importación de bibliotecas necesarias
import numpy as np
import pandas as pd
import shap
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

class SHAPExplainer:
    """
    Clase que facilita la carga de datos, entrenamiento de modelos,
    cálculo y visualización de valores SHAP para interpretabilidad de modelos de aprendizaje automático.
    """

    def __init__(self, csv_path, model_class, target_column, n_splits=10, test_size=0.3, random_state=42):
        """
        Inicializa el explicador SHAP.

        Parámetros:
        - csv_path: ruta del archivo CSV que contiene los datos.
        - model_class: clase del modelo (por ejemplo, RandomForestClassifier).
        - target_column: nombre de la columna objetivo.
        - n_splits: número de divisiones para validación cruzada.
        - test_size: proporción de datos reservados para prueba.
        - random_state: semilla para reproducibilidad.
        """
        self.csv_path = csv_path
        self.model_class = model_class
        self.target_column = target_column
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.modelos_guardados = []
        self.resultados = []
        self.shap_values = None
        self.modelo_entrenado_70 = None
        self._load_data()

    def _load_data(self):
        """
        Carga el conjunto de datos desde el archivo CSV, separa X e y,
        normaliza los datos de entrada y realiza la partición en entrenamiento y prueba.
        """
        df = pd.read_csv(self.csv_path)
        self.df = df.copy()
        self.y = df[self.target_column].values
        self.X = df.drop(columns=[self.target_column]).values
        self.X_norm = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_norm, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state)

    def train_and_evaluate(self):
        """
        Entrena el modelo utilizando validación cruzada estratificada y guarda los resultados de precisión (accuracy).
        Finalmente, entrena un modelo final con el 70% de los datos.
        """
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (train_index, val_index) in enumerate(cv.split(self.X_train, self.y_train), start=1):
            X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
            y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]
            modelo = self.model_class(random_state=self.random_state)
            modelo.fit(X_fold_train, y_fold_train)
            self.modelos_guardados.append(modelo)
            accuracy = modelo.score(X_fold_val, y_fold_val)
            self.resultados.append({'Pliegue': f'Pliegue {fold}', 'Accuracy': accuracy})

        print("\nResultados de Validación Cruzada:")
        print(pd.DataFrame(self.resultados))

        # Entrena el modelo final sobre el 70% de los datos
        self.modelo_entrenado_70 = self.model_class(random_state=self.random_state)
        self.modelo_entrenado_70.fit(self.X_train, self.y_train)
        print("\nModelo final entrenado sobre el 70% de los datos.")

    def calculate_shap_values(self):
        """
        Calcula los valores SHAP sobre los datos de prueba utilizando el modelo entrenado.
        """
        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de calcular los valores SHAP.")
        explainer = shap.TreeExplainer(self.modelo_entrenado_70)
        self.shap_values = explainer.shap_values(self.X_test)
        print("Valores SHAP calculados con forma:", np.array(self.shap_values).shape)

    def save_shap_values(self, filename="shap_values.pickle"):
        """
        Guarda los valores SHAP en un archivo pickle.

        Parámetros:
        - filename: nombre del archivo donde se guardarán los valores SHAP.
        """
        if self.shap_values is None:
            raise ValueError("Debe calcular los valores SHAP antes de guardarlos.")
        with open(filename, 'wb') as handle:
            pickle.dump(self.shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Valores SHAP guardados en {filename}")

    def explain_instance(self, instance, instance_type="new", class_names=None, instance_label="Instancia",
                         all_pairs=False):
        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de usar SHAP.")

        if instance_type == "test":
            if isinstance(self.shap_values, list):
                shap_values_instance = [shap_class[instance] for shap_class in self.shap_values]
            else:
                shap_values_instance = [self.shap_values[instance, :, i] for i in range(self.shap_values.shape[2])]
            probabilities_instance = self.modelo_entrenado_70.predict_proba([self.X_test[instance]])[0]
        else:
            instance_scaled = self.scaler.transform([instance])
            explainer = shap.TreeExplainer(self.modelo_entrenado_70)
            shap_values_raw = explainer.shap_values(instance_scaled)
            probabilities_instance = self.modelo_entrenado_70.predict_proba(instance_scaled)[0]
            if isinstance(shap_values_raw, list):
                shap_values_instance = [shap_values_raw[i][0] for i in range(len(shap_values_raw))]
            else:
                shap_values_instance = [shap_values_raw[0, :, i] for i in range(shap_values_raw.shape[2])]

        num_classes = len(shap_values_instance)
        classes = [class_names[i] if class_names else f"Clase {i}" for i in range(num_classes)]

        if num_classes < 2:
            print("Advertencia: Solo se detectó una clase. No se puede comparar entre clases.")
            return

        differences_shap_list = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diff = abs(probabilities_instance[i] - probabilities_instance[j])
                shap_differences = shap_values_instance[i] - shap_values_instance[j]
                differences_shap_list.append((diff, classes[i], classes[j], shap_differences))

        differences_shap_list_sorted = sorted(differences_shap_list, key=lambda x: x[0], reverse=True)

        # Gráfico 1: Diferencias de probabilidad
        plt.figure(figsize=(16, 12))
        sns.barplot(
            x=[diff[0] for diff in differences_shap_list_sorted],
            y=[f"{diff[1]} vs {diff[2]}" for diff in differences_shap_list_sorted],
            hue=[f"{diff[1]} vs {diff[2]}" for diff in differences_shap_list_sorted],
            palette="viridis",
            legend=False
        )
        plt.title(f"Diferencias Absolutas entre Probabilidades - {instance_label}", fontsize=22)
        plt.xlabel(r"Diferencias Absolutas $|\mathrm{\Delta P}|$", fontsize=20)
        plt.ylabel("Pares de Clases", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()

        # Gráficos SHAP por par de clases
        pares_a_graficar = differences_shap_list_sorted if all_pairs else [differences_shap_list_sorted[0]]
        for diff, class_1, class_2, shap_differences in pares_a_graficar:
            shap_diff_df = pd.DataFrame({
                "Feature": self.df.drop(columns=[self.target_column]).columns,
                "SHAP_Difference": shap_differences
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
                f"Diferencias en Valores SHAP\nEntre Clases '{class_1}' y '{class_2}' - {instance_label}",
                fontsize=26
            )
            plt.xlabel(r"Diferencias SHAP $\mathrm{\Delta S}$", fontsize=24)
            plt.ylabel("Características en orden descendente", fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.subplots_adjust(left=0.35)
            plt.tight_layout()
            plt.show()

    def get_probability_differences(self, instance, instance_type="test", class_names=None):
        """
        Calcula las diferencias absolutas de probabilidad entre pares de clases para una instancia dada.
        Devuelve una lista ordenada [(diferencia, clase_i, clase_j), ...]
        """
        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de usar SHAP.")

        # Obtener probabilidades de la instancia
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

        return sorted(differences, key=lambda x: x[0], reverse=True)

    def explain_selected_pairs(self, instance, selected_pairs, instance_type="test",
                               class_names=None, instance_label="Instancia"):
        """
        Explica diferencias SHAP para pares de clases seleccionados por el usuario.

        Parámetros:
        - instance: vector o índice de instancia.
        - selected_pairs: lista de tuplas con nombres de clases, ej. [("setosa", "versicolor")].
        - instance_type: "test" o "new".
        - class_names: diccionario opcional de nombres de clase.
        - instance_label: texto para mostrar en los gráficos.
        """
        if self.modelo_entrenado_70 is None:
            raise ValueError("Debe entrenar el modelo antes de usar SHAP.")

        # Obtener SHAP values y probabilidades
        if instance_type == "test":
            if isinstance(self.shap_values, list):
                shap_values_instance = [shap_class[instance] for shap_class in self.shap_values]
            else:
                shap_values_instance = [self.shap_values[instance, :, i] for i in range(self.shap_values.shape[2])]
            probabilities_instance = self.modelo_entrenado_70.predict_proba([self.X_test[instance]])[0]
        else:
            instance_scaled = self.scaler.transform([instance])
            explainer = shap.TreeExplainer(self.modelo_entrenado_70)
            shap_values_raw = explainer.shap_values(instance_scaled)
            probabilities_instance = self.modelo_entrenado_70.predict_proba(instance_scaled)[0]
            if isinstance(shap_values_raw, list):
                shap_values_instance = [shap_values_raw[i][0] for i in range(len(shap_values_raw))]
            else:
                shap_values_instance = [shap_values_raw[0, :, i] for i in range(shap_values_raw.shape[2])]

        # Diccionario inverso de class_names para encontrar índices
        inverse_class_names = {v: k for k, v in class_names.items()} if class_names else {f"Clase {i}": i for i in
                                                                                          range(
                                                                                              len(probabilities_instance))}

        # Graficar pares seleccionados
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
            plt.show()

    def plot_probability_differences(self, instance, instance_type="test", class_names=None,
                                     instance_label="Instancia"):
        """
        Grafica las diferencias absolutas de probabilidad entre pares de clases para una instancia.
        """
        diffs = self.get_probability_differences(instance, instance_type=instance_type, class_names=class_names)

        plt.figure(figsize=(16, 10))
        labels = [f"{diff[1]} vs {diff[2]}" for diff in diffs]
        sns.barplot(
            x=[diff[0] for diff in diffs],
            y=labels,
            hue=labels,  # <-- Asigna 'hue' para cumplir la nueva convención
            palette="viridis",
            legend=False
        )
        plt.title(f"Diferencias Absolutas entre Probabilidades - {instance_label}", fontsize=22)
        plt.xlabel(r"Diferencias Absolutas $|\mathrm{\Delta P}|$", fontsize=20)
        plt.ylabel("Pares de Clases", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

        return diffs  # También retorna la lista ordenada si se desea imprimir
