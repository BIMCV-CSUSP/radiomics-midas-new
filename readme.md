# Towards Computer-Aided Assessment of Lumbar Disc Degeneration Based on Radiomics

## Authors
Alejandro Mora-Rubio¹ [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0001-6012-8645), 
Jesús Alejandro Alzate-Grizales¹ [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0003-1021-2050), 
Joaquim Montell Serrano¹ [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0002-5597-7236), 
Carlos Mayor-deJuan², 
Rafael Llombart Blanco² [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0002-8790-0428), 
Mariola Penadés Fons³ [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0002-7299-7802), and 
Maria de la Iglesia-Vayá¹³ [<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" 
alt="ORCID logo" width="16" height="16" />](https://orcid.org/0000-0003-4505-8399)

¹ Unidad Mixta de Imagen Biomédica FISABIO-CIPF, Fundación para el Fomento de la Investigación Sanitario y Biomédica de la Comunidad Valenciana, Valencia, Spain 

² Clínica Universidad de Navarra, Pamplona, Spain

³ Dirección General de Investigación e Innovación, Conselleria de Sanitat, Valencia, Spain

## Abstract
Intervertebral disc degeneration (IDD) is a common age-related condition characterized by structural alterations and functional impairment of intervertebral discs, leading to various symptoms such as back and neck pain. The Pfirrmann grading system is commonly used for assessing IDD severity based on Magnetic Resonance Imaging (MRI) findings. However, this method is only qualitative, resulting in high inter-rater variability, and also time-consuming. In this study, we aimed to develop a machine learning model to automate Pfirrmann grade classification of IDD using quantitative radiomic features extracted from MRI scans. We retrospectively collected 717 MRI scans, which were manually labeled by an expert radiologist. The extracted features were utilized to train different classifiers, achieving for the 5 level grading task an average accuracy of 71\%, F1 score of 70\%, and without any disc being graded outside a $\pm1$ margin. Our findings demonstrate the potential of radiomics and machine learning to automate IDD assessment, reduce inter-rater variability, improve clinical efficiency, and support more accurate diagnoses and personalized treatment plans.

## About this repo
1. **figures/**: Visual outputs and plots from the radiomics analysis.
  - `imageslabel1/`: Contains 20 subfolders (one per top feature), each with a violin plot (`violinplot.png`) showing the distribution of that feature across Pfirrmann classes.

2. `**notebooks/**: contains Jupyter Notebooks, used during the experimentation phase.
    -`EDA_image` : basic exploratory analysis of lumbar MRI images, including image size, spacing, and visualization of images and masks.
    -`Image_Pre_processing`: basic MRI image pre-processing
    -`extract_radiomics`: 1. Extraction and merging of radiomic features from lumbar MRI images using PyRadiomics.
    2.Statistical analysis and machine learning modeling for Pfirrmann grade classification.
    -`train_and_evaluate`: selection of relevant radiomic features and evaluation of multiple machine learning models for Pfirrmann grade classification.

3. Source (`src`): contains the final Python scripts.
    `Params.yaml` congfiguration file.
## Publication


### Cite as
