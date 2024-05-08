# Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics

## Authors
Alejandro Mora-Rubio¹ [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0001-6012-8645), 
Jesús Alejandro Alzate-Grizales¹ [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0003-1021-2050), 
Joaquim Montell Serrano¹ [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0002-5597-7236), 
Carlos Mayor-deJuan² [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/), 
Rafael Llombart Blanco² [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0002-8790-0428), 
Mariola Penadés Fons³ [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0002-7299-7802), and 
Maria de la Iglesia-Vayá¹³ [![ORCID favicon](https://orcid.figshare.com/ndownloader/files/8439032)](https://orcid.org/0000-0003-4505-8399)

¹ Unidad Mixta de Imagen Biomédica FISABIO-CIPF, Fundación para el Fomento de la Investigación Sanitario y Biomédica de la Comunidad Valenciana, Valencia, Spain 

² Clínica Universidad de Navarra, Pamplona, Spain

³ Dirección General de Investigación e Innovación, Conselleria de Sanitat, Valencia, Spain

## Abstract
Intervertebral disc degeneration (IDD) is a common age-related condition characterized by structural alterations and functional impairment of intervertebral discs, leading to various symptoms such as back and neck pain. The Pfirrmann grading system is commonly used for assessing IDD severity based on Magnetic Resonance Imaging (MRI) findings. However, this method is only qualitative, resulting in high inter-rater variability, and also time-consuming. In this study, we aimed to develop a machine learning model to automate Pfirrmann grade classification of IDD using quantitative radiomic features extracted from MRI scans. We retrospectively collected 717 MRI scans, which were manually labeled by an expert radiologist. The extracted features were utilized to train different classifiers, achieving for the 5 level grading task an average accuracy of 71\%, F1 score of 70\%, and without any disc being graded outside a $\pm1$ margin. Our findings demonstrate the potential of radiomics and machine learning to automate IDD assessment, reduce inter-rater variability, improve clinical efficiency, and support more accurate diagnoses and personalized treatment plans.

## About this repo
1. `figures`: visual outputs or graphs related to the radiomics analysis.
2. `notebooks`: contains Jupyter Notebooks, used during the experimentation phase.
    - `pyradiomics` refers to the feature extraction process.
    - `radiomics-analysis` contains the whole machine learning pipeline, from EDA to results evaluation.
    - `cv-analysis` [UNDER DEVELOPMENT] refers to preliminary experiments on IDD classification using Computer Vision.
3. Source (`src`): contains the final Python scripts.
    - `radiomics/calculate` executes the feature extraction process using the provided `Params.yaml` congfiguration file.
    - `ml/test_multiple_models` evaluates a suite of machine learning models for each experiment.
    - `ml/random_search` evaluates different parameter combinations for the best model for each experiment.
    - `ml/compute_metrics` evaluates the best performing model for each experiment using cross-validation.
    - `ml/transforms` and `ml/utils` contain helper functions for data import and preprocessing.

## Publication


### Cite as