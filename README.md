# Lumbar Intervertebral Disc Degeneration: Pfirrmann Classification
This repository contains an end-to-end machine learning pipeline for the automated classification of lumbar intervertebral disc degeneration based on the Pfirrmann grading system.

The project workflow goes from raw medical images to a fully trained, statistically validated, and explainable machine learning model.


## 1. Project Pipeline
The methodology is divided into three sequential stages:

- Total Segmentator (Mask Extraction): The first step involves processing the raw medical images to isolate the regions of interest. We extract the intervertebral disc masks, order them sequentially (from disc 1 to 5), and filter the data to maintain only these 5 specific lumbar disc masks per patient.

- Radiomics Extraction: Once the masks are isolated, we analyze different imaging modalities to extract quantitative radiomic features from each disc. These features are then merged into a single structured CSV file containing all the radiomics data for each disc in every patient.

- Machine Learning Classification: The final step is training and evaluating predictive models based on the extracted radiomics. This stage involves an automated pipeline that:

  - Cleans the data, performs statistical feature selection, and trains multiple baseline models using Stratified Group K-Fold cross-validation.

  - Evaluates the statistical significance between model performances using Friedman and Wilcoxon tests.

  - Fine-tunes the best-performing model using Bayesian optimization, calibrates its probabilities, and provides interpretability using SHAP values.


## 2. Repository Branches
To keep the development organized, the code and experimental results are divided into specific branches. Please switch to the relevant branch depending on the part of the pipeline you want to explore:

- total_segmentator
Contains the scripts and instructions for the initial image processing. It covers the extraction of the intervertebral disc masks, the naming/ordering logic (1 to 5), and the filtering process to retain only the necessary masks.

- radiomics_extraction
Contains the radiomics_extraction.py script and related tools. This branch handles the extraction of radiomic features across different image modalities and the generation of the consolidated CSV dataset. (See the README inside this branch for detailed execution instructions).

- first_results_ML
Contains the code and results for the initial machine learning experiments. These models were trained on an early, more unbalanced dataset to establish a baseline for Pfirrmann disc degeneration classification.

- second_results_ML
Contains the updated and most robust machine learning experiments. This branch uses a newer, expanded dataset with a larger number of patients, specifically balancing the data by including more samples of Pfirrmann grades 1 and 5.

🛠️ Getting Started
