# Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics

---

## 👩‍🔬 Authors

*Add author names and affiliations here.*

---

## 📝 Abstract

*Provide a brief summary of the project, objectives, and main findings.*

---

## 📂 About this Repository

This repository contains code, notebooks, and configuration files for a radiomics-based pipeline to classify lumbar intervertebral disc degeneration (Pfirrmann grading) from MRI-derived radiomic features.

---
## 📁 High-Level Repository Structure

```
├── data/                     # Input features, intermediate artifacts, predictions & metrics
│   ├── binary/               # Binary classification outputs (e.g. healthy vs degenerated)
│   ├── multiclass/           # Multiclass (full Pfirrmann grades) outputs
│   ├── datos_EDA/            # Raw exploratory data analysis (EDA) exports
│   └── ...                   # Feature selection, model differences, ROC curves, etc.
├── images_EDA/               # Generated exploratory plots (image-related + NLP wordclouds)
├── notebooks/                # Jupyter notebooks and equivalent Python scripts
├── results/                  # Final aggregated performance summaries (exported figures/tables)
├── src/                      # (Placeholder) Core reusable Python modules (if added later)
├── Params.yaml               # Global configuration (can include feature sets / paths)
├── Params_alldisc.yaml       # Alternative configuration (e.g. all-disc setting)
├── readme.md                 # Project documentation
```

---
## 📦 Folder & File Details

### data/
Stores input CSV feature matrices and generated artifacts after running training scripts.

Key subfolders (illustrative):
- binary/:
  - features_t2w_BPfirrmann.csv: Feature matrix for binary task.
  - preds_lumbar_discs_label.csv: Serialized per-fold predictions.
  - results_lumbar_discs_label.csv: Metrics per fold & model.
  - variables_used.txt: Final feature subset used in training.
  - feature_selection/: CSVs & plots for selected features.
  - model_differences/: Statistical comparison outputs (p-values, adjusted tests, boxplots).
  - ROC_curves/: ROC plots per model/fold.
- multiclass/:
  - features_t2w_MPfirrmann.csv: Multiclass feature matrix.
  - preds_discoslumbar.csv / resultados_discoslumbar.csv: Prediction + metric exports.
  - variables_used.txt: Selected features.
  - (Mirrors binary/ structure for multiclass scenario.)
- datos_EDA/: Auxiliary exports produced during data/image exploratory analysis.

### images_EDA/
High-resolution plots (PNG/PDF) from image and textual EDA:
- distribution_pfirrmann_grades_by_disc.*: Class distribution per disc level.
- espaciado_t2w.*, tamaño_t2w.*: Image spacing & size statistics.
- image_preprocessing.*, image_mask_example.*: Preprocessing and mask visualization.
- wordcloud_*.png/pdf: Token frequency visualizations from clinical descriptions.

### notebooks/
Interactive development + reproducible analysis. Some scripts have .py clones for headless runs.

Core notebooks / scripts:
- EDA_image.ipynb: Image geometry & quality assessment (spacing, size, missingness checks).
- EDA-variables.ipynb: Statistical profiling of radiomic features (distributions, correlations).
- Image_Pre_processing.ipynb: Pre-processing pipeline overview (resampling, normalization, masking).
- extract_radiomics.ipynb: PyRadiomics feature extraction & harmonization (merging patient metadata).
- extract_patients.ipynb: Patient / study filtering and cohort assembly.
- train_and_evaluate.ipynb / train_and_evaluate_copy.ipynb: Binary classification training loop (feature selection + CV evaluation).
- best_model.ipynb: Inspection / interpretation of the selected best-performing model.
- shap_multiclass.ipynb: SHAP-based explainability of multiclass classifier.
- train_binary.py: Script version for binary task (CLI-friendly).
- train_multiclass2.py: End-to-end multiclass pipeline (feature selection, CV, ROC generation, statistical comparisons, optional fine-tuning).
- retrain_multiclass.py / 3_retrain_best_model_and_evaluate*.py: Retraining best model on full training data & final evaluation.
- 2_model_differences.py: Statistical comparison among models (e.g. pairwise tests on AUC across folds).

### results/
Post-processed performance summaries (aggregated tables, potentially publication-ready figures) separated into binary/ and multiclass/ contexts.

### src/
Reserved for modular code (e.g., utility functions, preprocessing abstractions). Currently minimal; logic mainly resides in notebooks / training scripts.

### Config Files
- Params.yaml / Params_alldisc.yaml: Store adjustable parameters (paths, feature subset identifiers, CV seeds). Encouraged to externalize hard-coded constants from scripts into these files for cleaner reproducibility.

---
## 🔁 Processing & Modeling Workflow

1. Data acquisition & curation (extract_patients.ipynb).
2. MRI pre-processing (Image_Pre_processing).* (resampling, masking, intensity normalization).
3. Feature extraction (extract_radiomics.ipynb) via PyRadiomics.
4. Exploratory data analysis (EDA_image / EDA-variables) to assess distributions & correlations.
5. Feature selection (inside train_* scripts):
   - Normality test (Shapiro) per feature.
   - If normal: ANOVA; else: Kruskal–Wallis.
   - Sort by p-value; retain at most floor(N_samples / 15) top features.
6. Model training & evaluation:
   - Repeated StratifiedGroupKFold (patient grouping) to avoid leakage.
   - Metrics per fold: AUC (macro OvR), F1 macro, accuracy, balanced accuracy, Cohen's Kappa, MCC, per-class precision/recall/F1/accuracy.
7. ROC curve generation:
   - For each classifier: optimal fold (max validation AUC) & median fold.
   - One-vs-Rest ROC per class saved (PNG/PDF).
8. Model statistical comparison (2_model_differences.py):
   - Pairwise tests across model AUC distributions.
   - Multiple testing correction.
9. (Optional) Fine-tuning best model: Invokes train_multiclass.py with selected feature subset.
10. Interpretation (shap_multiclass.ipynb): Global & local importance explanations.

---
## 🧪 Cross-Validation Design
- StratifiedGroupKFold: Preserves label distribution across folds while grouping by patient_id.
- Repeats: Increases robustness by re-shuffling (base_random_state + repetition index).
- Aggregation: Mean AUC used to select best model; fold-level metrics retained for statistical inference.

---
## 📊 Key Output Files

Inside each experiment directory (e.g., data/features_t2w_multiclass/):
- resultados_discoslumbar.csv: Fold-level metrics per classifier.
- preds_discoslumbar.csv: Serialized predictions (y_true, y_pred, probability matrix).
- variables_used.txt: Final feature names after selection.
- ROC_curves/*.png|pdf: OvR ROC curves (optimal & median fold).
- feature_selection/train_auc_pvals_df.csv: Statistical test + p-values (subset displayed if filtered).
- model_differences/: Statistical test summaries & significance plots.

---
## 🏃 Running the Pipelines

Example (multiclass):
```
python notebooks/train_multiclass2.py \
  --csv features_t2w_MPfirrmann.csv \
  --n_splits 5 \
  --n_repeats 10 \
  --feature_strategy most_discriminant \
  --fine_tune_best_model
```

Binary (analogous):
```
python notebooks/train_binary.py --csv features_t2w_BPfirrmann.csv
```

Notes:
- Ensure paths inside scripts match your environment (consider refactoring into Params.yaml).
- Run from project root so relative script paths resolve.

---
## 🧬 Feature Selection Rationale
Restricting features to ~ N/15 guards against overfitting with limited sample size, balancing model complexity and variance. Non-parametric tests are used when normality is rejected to maintain robustness.

---
## 📐 Statistical Model Comparison
`2_model_differences.py` consumes:
- Predictions CSV (per-fold probability outputs)
- Metrics CSV (per-fold AUC values)
Performs pairwise comparisons (e.g., Wilcoxon / Mann–Whitney depending on design—inspect script) and applies multiple hypothesis correction (e.g., Holm / FDR). Outputs adjusted p-values and significance tables.

---
## 📄 Publication

*Add publication details or links here.*

---
## 📚 Cite as

*Provide citation information here.*

---
## 🙌 Acknowledgments
List funding sources, institutional support, and open-source tools (PyRadiomics, scikit-learn, matplotlib, seaborn, SHAP, etc.).

---
## 📝 License
*Specify license (e.g., MIT, Apache-2.0) if applicable.*

