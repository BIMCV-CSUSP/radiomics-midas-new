# Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics

---

## 👩‍🔬 Authors

*Add author names and affiliations here.*

---

## 📝 Abstract

*Provide a brief summary of the project, objectives, and main findings.*

---

## 📂 About this Repository

This repository contains code and resources for the radiomics-based assessment of lumbar disc degeneration.

### 📁 Repository Structure

```
├── figures/
│   └── imageslabel1/
│       └── [20 subfolders: one per top feature]
│           └── violinplot.png
├── notebooks/
│   ├── EDA_image.ipynb
│   ├── Image_Pre_processing.ipynb
│   ├── extract_radiomics.ipynb
│   └── train_and_evaluate.ipynb
├── src/
├── Params.yaml
```

- **figures/**  
  Visual outputs and plots from the radiomics analysis.
  - `imageslabel1/`: 20 subfolders (one per top feature), each with a violin plot (`violinplot.png`) showing feature distribution across Pfirrmann classes.

- **notebooks/**  
  Jupyter Notebooks used during experimentation:
  - `EDA_image`: Exploratory analysis of lumbar MRI images (size, spacing, visualization).
  - `Image_Pre_processing`: Basic MRI image pre-processing.
  - `extract_radiomics`: Radiomic feature extraction/merging with PyRadiomics, statistical analysis, and ML modeling for Pfirrmann grade classification.
  - `train_and_evaluate`: Feature selection and evaluation of ML models for Pfirrmann grade classification.

- **src/**  
  Final Python scripts.

- **Params.yaml**  
  Configuration file for experiments and scripts.

---

## 📄 Publication

*Add publication details or links here.*

---

## 📚 Cite as

*Provide citation information here.*

