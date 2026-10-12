# Experiment artifacts

This directory contains the small artifacts required by the tabular and
Fashion-MNIST examples.

## Included in the repository

- `iris/shap_values_iris.pickle`
- `seeds/shap_values_seeds_lcd_mlp_probability.pickle`
- `seeds/shap_values_seeds_mlp_probability_0_vs_1.pickle`
- `vehicle/shap_values_vehicle_dt.pickle`
- `wine/shap_values_wine_xgb_probability.pickle`
- `fashion_mnist/cnn_fashion.keras`
- `fashion_mnist/shap_values_fashion_paper_M10.pickle`

## External artifacts

The following artifacts are not tracked by Git because of their size.
They must be placed at the exact paths shown below before running the
corresponding examples.

### HAM10000

- `ham10000/ham10000_resnet50_best_overall_lesion_split_030926.weights.h5`
- `ham10000/ham10000_split_manifest_by_lesion_030926.csv`
- `ham10000/shap_values_ham10000_resnet50_lesion_split_030926_1_vs_2.pickle`

### BloodMNIST

- `bloodmnist/bloodmnist_resnet18_best.keras`
- `bloodmnist/shap_values_bloodmnist_resnet18_best_6_vs_1.pickle`

### AG News

Place the complete trained model directory at:

- `ag_news/model_agnews_distilbert_base_uncased/`

It must contain at least `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `model.safetensors`.

### 20 Newsgroups

Place the trained model and saved test information at:

- `20_newsgroups/model_20newsgroups_roberta_base/`
- `20_newsgroups/test_split.csv`
- `20_newsgroups/test_predictions.csv`
- `20_newsgroups/label_mapping.csv`

### Child occupation

Place the definitive validated artifact package at:

- `child_occupation/`

This package includes the trained model, preprocessing pipeline, scalers,
train/test partitions, feature metadata, predictions, and SHAP values used
by the local explanation and robustness evaluation.

## Availability

Large artifacts will be distributed through an external research-data
archive. Its persistent download link will be added here when the archive
is published.
