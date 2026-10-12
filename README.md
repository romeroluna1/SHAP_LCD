# SHAP-LCD

**Local explanations for class differentiation in multiclass classification models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![DOI](https://img.shields.io/badge/DOI-10.1002%2Faisy.70397-blue.svg)](https://doi.org/10.1002/aisy.70397)

SHAP-LCD explains why a model favors one class over another for a particular instance. Given two classes, $C_j$ and $C_r$, it compares their predicted probabilities and their class-specific SHAP contributions:

$$
\Delta P_i^{(C_j,C_r)} = P_i(C_j)-P_i(C_r)
$$

$$
\Delta S_{m,i}^{(C_j,C_r)}
= s_{m,i}^{(C_j)}-s_{m,i}^{(C_r)}
$$

A positive feature-level difference favors $C_j$, whereas a negative difference favors $C_r$.

## Repository contents

```text
SHAP_LCD/
|-- src/shap_lcd/            Core SHAP-LCD implementation
|-- examples/
|   |-- tabular/             Iris, Wine, Vehicle, Seeds, child occupation
|   |-- image/               Fashion-MNIST, HAM10000, BloodMNIST
|   \-- text/                AG News and 20 Newsgroups
|-- evaluation/tabular/      Robustness evaluation in original feature space
|-- data/                    Small tabular datasets
|-- artifacts/               Included artifacts and external-artifact guide
|-- results/                 Selected final and local results
|-- requirements.txt         Core and tabular dependencies
|-- requirements-image.txt   Additional image dependencies
\-- requirements-text.txt    Additional text dependencies
```

Historical root-level scripts remain available through the Git history. The organized examples under `examples/` are the maintained versions.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/romeroluna1/SHAP_LCD.git
cd SHAP_LCD
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies needed for tabular examples:

```bash
python -m pip install -r requirements.txt
```

For image examples:

```bash
python -m pip install -r requirements-image.txt
```

For text examples:

```bash
python -m pip install -r requirements-text.txt
```

## Quick start

Run these commands from the repository root.

### Tabular examples

```bash
python examples/tabular/run_iris.py
python examples/tabular/run_wine.py
python examples/tabular/run_vehicle.py
python examples/tabular/run_seeds.py
python examples/tabular/run_child_occupation.py
```

### Image examples

```bash
python examples/image/run_fashion_mnist.py
python examples/image/run_ham10000.py
python examples/image/run_bloodmnist_lcd.py
```

### Text examples

```bash
python examples/text/run_ag_news.py
python examples/text/run_20_newsgroups.py
```

## Artifacts

Small artifacts required by the tabular and Fashion-MNIST examples are included. Large trained models, cached SHAP arrays, and research datasets are not stored directly in Git because of GitHub file-size limits.

See [`artifacts/README.md`](artifacts/README.md) for the required filenames and directory locations.

## Reproducibility and evaluation

The maintained examples use fixed random seeds where supported.

For tabular experiments that include model assessment, preprocessing is fitted inside each training fold during cross-validation. The definitive scaler and model are fitted only on the final training partition. This avoids using validation or test information when fitting preprocessing transformations.

The repository also verifies, where applicable, that SHAP values are expressed on the probability scale by checking the additive reconstruction:

$$
\text{base value}+\sum_m s_{m,i}^{(C_k)}
\approx P_i(C_k)
$$

The robustness evaluation for the child-occupation case perturbs variables in their original semantic space:

- numerical variables: bounded Gaussian perturbation;
- binary variables: valid binary changes;
- categorical variables: valid category substitutions.

The perturbed observations are subsequently processed by the same fitted training pipeline.

## Research examples

The repository contains:

1. Reproducibility examples associated with the SHAP-LCD methodological work.
2. Extended tabular, image, and text examples developed for subsequent research and monograph documentation.

Quick-test results are intended only to verify execution and must not be reported as final experimental results.

## Public API

```python
from shap_lcd import SHAPExplainer
```

Image and text utilities are available from their corresponding modules:

```python
from shap_lcd.image import SHAPExplainer2
from shap_lcd.text import TextSHAPLCDExplainer
```

## Citation

If you use SHAP-LCD, please cite:

> R. M. Romero Luna et al., "Shapley Additive Explanation for Local Class Differentiation: Local Explainability for Class Differentiation in Classification Models," *Advanced Intelligent Systems*. https://doi.org/10.1002/aisy.70397

## Authors

- Roxana M. Romero Luna - Universidad del Cauca
- Hugo A. Ordo&ntilde;ez
- Carlos A. Cobos

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Contact

Questions and suggestions may be submitted through GitHub Issues or sent to romeroluna@unicauca.edu.co.
