## Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics

This repository contains code and resources for radiomics-based assessment of lumbar disc degeneration (binary and multiclass Pfirrmann grading). It provides data preparation, exploratory analysis, feature extraction, model training/evaluation, statistical comparisons, and explainability (SHAP).

- Python ≥ 3.11 (used in code)
- Main libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, shap, lime, PyRadiomics (for feature extraction), SimpleITK, scikit-image

---

## Repository Structure

- [.gitignore](.gitignore)
- [readme.md](readme.md) – This file (project overview, structure, how to run).
- [src/](src)
  - [Params.yaml](src/Params.yaml) – Optional configuration (paths/experiment params) for radiomics.
- [code/](code)
  - Extraction radiomic features
    - [extract_radiomics.ipynb](code/extract_radiomics.ipynb) – Radiomic feature extraction/merge (PyRadiomics).
  - Training and evaluation
    - [train_binary.py](code/train_binary.py) – Binary classification training/evaluation (cross-validation, ROC, outputs to data/binary).
    - [train_multiclass.py](code/train_multiclass2.py) – Multiclass Pfirrmann training/evaluation (one-vs-rest ROC).
    - [2_model_differences.py](code/2_model_differences.py) – Statistical comparison across classifiers; produces summary + boxplot + p-value heatmap.
    - [3_retrain_best_model_and_evaluate_binary.py](code/3_retrain_best_model_and_evaluate.py) – Retrains best binary model; SHAP explainability and reports.
    - [3_retrain_best_model_and_evaluate_multiclass.py](code/3_retrain_best_model_and_evaluate_multiclass.py) – Same for multiclass; per-class SHAP beeswarm/heatmaps
- [data/](data)
  - Binary workflow outputs: [data/binary/](data/binary)
    - features_t2w_BPfirrmann.csv – Training features (T2w; binary Pfirrmann label).
    - results_lumbar_discs_label.csv – Per-fold metrics (e.g., val AUC, accuracy).
    - preds_lumbar_discs_label.csv – Per-fold predictions/probabilities.
    - variables_used.txt – Selected features (and ranking/selection context).
    - [feature_selection/](data/binary/feature_selection) – Selection logs, ranked features, and images (per-feature plots).
    - [ROC_curves/](data/binary/ROC_curves) – ROC plots for optimal/median folds.
    - [model_differences/](data/binary/model_differences) – Statistical comparisons (boxplot and p-value heatmap).
  - Multiclass workflow outputs: [data/multiclass/](data/multiclass)
    - features_t2w_MPfirrmann.csv – Training features (T2w; multiclass Pfirrmann).
    - resultados_discoslumbar.csv – Per-fold metrics (one-vs-rest AUCs, etc.).
    - preds_discoslumbar.csv – Per-fold predictions/probabilities.
    - variables_used.txt – Selected features used in modeling.
    - [feature_selection/](data/multiclass/feature_selection) – Selection artifacts.
    - [ROC_curves/](data/multiclass/ROC_curves) – Multiclass ROC (OvR) plots.
    - [model_differences/](data/multiclass/model_differences) – Statistical comparisons.
- [results/](results)
    - [results/binary/best_results/](results/binary/best_results) SHAP plots and confusion matrix.
    - [results/multiclass/best_results/](results/multiclass/best_results) Same for multiclass.

---

## Workflow

1) Feature extraction
- Extract radiomics with [code/extract_radiomics.ipynb](code/extract_radiomics.ipynb).
- Resulting CSVs are stored under [data/binary/](data/binary) and/or [data/multiclass/](data/multiclass).

2) Train and evaluate models
- Binary:
  - Run: `python code/train_binary.py`
  - Outputs:
    - Metrics: [data/binary/results_lumbar_discs_label.csv](data/binary/results_lumbar_discs_label.csv)
    - Predictions: [data/binary/preds_lumbar_discs_label.csv](data/binary/preds_lumbar_discs_label.csv)
    - ROC curves: [data/binary/ROC_curves/](data/binary/ROC_curves)
    - Selected features: [data/binary/variables_used.txt](data/binary/variables_used.txt)
- Multiclass:
  - Run: `python code/train_multiclass.py`
  - Outputs:
    - Metrics: [data/multiclass/resultados_discoslumbar.csv](data/multiclass/resultados_discoslumbar.csv)
    - Predictions: [data/multiclass/preds_discoslumbar.csv](data/multiclass/preds_discoslumbar.csv)
    - ROC curves (OvR): [data/multiclass/ROC_curves/](data/multiclass/ROC_curves)
    - Selected features: [data/multiclass/variables_used.txt](data/multiclass/variables_used.txt)

  2.1) Statistical comparison across classifiers
  - This is call inside each train.py [code/2_model_differences.py](code/2_model_differences.py):
    - Produces:
      - Summary text (global test + pairwise tests with Holm correction)
      - Boxplot: data/.../model_differences/boxplot_metric.png
      - P-value heatmap: data/.../model_differences/heatmap_pvalues.png

  2.2) Retrain best model and explainability
  This is call inside each train.py
  - Binary: [code/3_retrain_best_model_and_evaluate.py](code/3_retrain_best_model_and_evaluate.py)
  - Multiclass: [code/3_retrain_best_model_and_evaluate_multiclass.py](code/3_retrain_best_model_and_evaluate_multiclass.py)
  - Saves:
    - SHAP heatmaps/beeswarms per class
    - LIME explanations (pseudo-beeswarm and instance-level figures)
    - Reports under results/.../best_results/

This is the workflow followed in this project:
![Flowchart](<images_EDA/Flowchart.drawio.png>)

---

## Key Outputs

- Per-fold metrics CSVs:
  - Binary: [data/binary/results_lumbar_discs_label.csv](data/binary/results_lumbar_discs_label.csv)
  - Multiclass: [data/multiclass/resultados_discoslumbar.csv](data/multiclass/resultados_discoslumbar.csv)
- Per-fold predictions CSVs:
  - Binary: [data/binary/preds_lumbar_discs_label.csv](data/binary/preds_lumbar_discs_label.csv)
  - Multiclass: [data/multiclass/preds_discoslumbar.csv](data/multiclass/preds_discoslumbar.csv)
- ROC plots:
  - Binary: [data/binary/ROC_curves/](data/binary/ROC_curves)
  - Multiclass: [data/multiclass/ROC_curves/](data/multiclass/ROC_curves)
- Model comparisons:
  - Binary: [data/binary/model_differences/](data/binary/model_differences)
  - Multiclass: [data/multiclass/model_differences/](data/multiclass/model_differences)
- Selected features: variables_used.txt in each data modality folder.
- Results SHAP Values:
  - Binary: [results/binary/explicability/train/SHAP/]
  - Multiclass: [results/multiclass/explicability/train/SHAP/]
---

## Citation

If you use this repository, please cite:

González Martínez, S., Alzate-Grisales, J. A., Montell Serrano, J., Francisco, J., Domenech, J., & de la Iglesia-Vayá, M. (2025). Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics. Unidad Mixta de Imagen Biomédica e Inteligencia Artificial CIPF-FISABIO & Clínica Universidad de Navarra.

