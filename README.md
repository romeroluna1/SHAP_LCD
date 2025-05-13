# SHAP_LCD
_Evaluación de las contribuciones de las características a la diferenciación de clases en modelos de clasificación usando valores SHAP_

[![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

---

## ✨ Descripción breve (ES)

**SHAP_LCD** implementa un marco matemático y práctico para:
1. Calcular la **variabilidad de las probabilidades** entre pares de clases de un modelo de clasificación y  
2. Relacionar esa variabilidad con las **contribuciones SHAP** de las características a nivel de instancia.

Esto permite conocer **qué atributos impulsan que una instancia cambie de una clase a otra**, proporcionando informes locales interpretables para científicos de datos, sociólogos y responsables de políticas públicas.

## ✨ Short description (EN)

**SHAP_LCD** provides a mathematical and practical framework to:
1. Compute the **probability variability** between pairs of classes in a classifier, and  
2. Link that variability to **feature‐level SHAP contributions** for each instance.

It helps you discover **which features drive an instance to switch from one class to another**, yielding local, human-readable reports.

---

## 📂 Estructura del repositorio / Repository layout

```
SHAP_LCD/
├── SHAPE_Explainer.py       # Clase principal que orquesta entrenamiento, cómputo y visualización
├── Main.py                  # Ejecución desde línea de comandos (ejemplo end-to-end)
├── Ejemplo_Iris.py          # Notebook / script de ejemplo con el dataset Iris
├── iris.csv                 # Conjunto de datos de demostración
├── shap_values_iris.pickle  # Valores SHAP pre-calculados (opcional)
└── requirements.txt         # Dependencias de Python  (añade si aún no existe)
```

---

## ⚙️ Instalación / Installation

> **Requisitos**  
> - Python ≥ 3.9  
> - `pip` o `conda`  
> - Compilador C/C++ si usas XGBoost u otras libs que lo necesiten

```bash
# 1. Clonar el proyecto
git clone https://github.com/romeroluna1/SHAP_LCD.git
cd SHAP_LCD

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt  # Añade aquí scikit-learn, shap, pandas, etc.
```

---

## 🚀 Uso rápido / Quick start

### 1. Desde un script de ejemplo

```bash
python Ejemplo_Iris.py
```

Generará:
- Diferencias absolutas de probabilidad para cada par de clases  
- Gráficos de barras y boxplots mostrando `ΔP` y `ΔSHAP`  
- Un reporte en `./reports/` (JSON + imágenes)

### 2. Con la clase `SHAPE_Explainer`

```python
from SHAPE_Explainer import SHAPEExplainer
from sklearn.ensemble import RandomForestClassifier

explainer = SHAPEExplainer(
    csv_path="iris.csv",
    model_class=RandomForestClassifier,
    target_column="species",
    n_splits=10,           # Validación cruzada estratificada
    test_size=0.3,
    random_state=42
)

explainer.fit()                                                # Entrena y guarda shap_values_
explainer.explain_instance(instance_id=0, top_k=10)            # Visualiza contribuciones
explainer.save_reports(output_dir="reports/iris_instance0")    # Exporta resultados
```

---

## 📊 Ejemplo de salida

![Sample ΔP vs ΔSHAP](docs/example_delta_plot.png) <!-- TODO: añade captura si lo deseas -->

---

## 🤝 Contribuciones / Contributing

1. Haz un fork y crea tu rama (`git checkout -b feature/mi-mejora`).  
2. Asegúrate de que `pre-commit` (black, flake8, isort) pasa sin errores.  
3. Envía un _pull request_ describiendo claramente la mejora o corrección.

---

## 📝 Licencia / License
Distribuido bajo la licencia **MIT**. Consulta el archivo [`LICENSE`](LICENSE) para más información.  
<!-- TODO: cámbiala si decides otra licencia -->

---

## 👥 Autores / Authors

| Nombre | Afiliación | Correo |
|--------|------------|--------|
| Roxana R. | [TODO] | [TODO] |
| Colaboradores | ¡Tu nombre aquí! | |

¿Usas **SHAP_LCD** en tu investigación? ¡Cítanos y abre un _issue_ para contarnos!  

---

## 📚 Referencias clave / Key references

- Lundberg, S. M., & Lee, S.-I. (2017). “A Unified Approach to Interpreting Model Predictions.”  
- [SHAP documentation](https://shap.readthedocs.io/en/latest/) (2025-01-08).  
- **[Añade aquí tu artículo o pre-print asociado]**

---

## ✉️ Contacto / Contact

Para dudas o sugerencias abre un **Issue** o escribe a **[correo-de-contacto]**.  
