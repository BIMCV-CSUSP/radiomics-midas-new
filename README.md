# Radiomics Extraction Pipeline
This branch contains the automated pipeline for extracting quantitative radiomic features from medical imaging modalities (MRI) and structuring them for Machine Learning. The system is built with a strict separation between the Logic Module and the Data Module to ensure modularity, reproducibility, and portability across different environments.
## File and Directory Architecture
### 1. Logic Module (Code + Configuration)
The following files must remain together in the script execution directory:

- radiomics_extraction.py: The main execution script.

- Params_template.yaml: The general PyRadiomics configuration template.

- Merge_radiomics_csv.ipynb: A Jupyter Notebook used for the final data restructuring and merging.

Note: The extraction script dynamically generates modality-specific configuration files (Params_<MODALITY>.yaml) by analyzing the statistical properties of the input images (e.g., adjusting bin width and spacing, disabling normalization for ADC).


### 2. Data Module (Input / Output)
The script processes an input CSV and automatically generates results in the same directory where the CSV is located.

- Output Directory: radiomic_results/ (Auto-generated next to the input CSV).

- Contents: Contains the extracted radiomics feature CSV files (separated by modality and extraction type) and the optimized YAML files used for the extraction.

## Data Requirements
The pipeline requires an input CSV file (data.csv) containing absolute paths to guarantee portability.

Required Columns:

- patient_id: Unique identifier for the patient.

- mask: Absolute path to the segmentation mask (.nii.gz).

- Modality Columns: At least one of the following must be present: T1, T1c, T2, ADC, DWI. The script automatically detects valid modalities by scanning the column headers.

Optional Columns:

- study_id

Note: If multiple modalities are present, all are processed in a single execution. If no valid modality columns are found, execution will stop with an error.

## Processing Workflow
The extraction pipeline utilizes a ProcessPoolExecutor to process patients sequentially within parallel workers, ensuring process isolation and preventing shared-memory conflicts.

1. Initial Data Analysis: Samples the dataset to calculate the Optimal Bin Width and Target Spacing for each modality. Custom YAMLs are generated.

2. Input Validation: Verifies the existence of required columns and checks that all referenced image and mask files exist on the disk.

3. Preprocessing: Each image undergoes N4 bias field correction and Curvature anisotropic diffusion denoising (performed in float32 precision).

4. Geometry Consistency Check: Verifies spatial consistency between the image and mask. If spacing, origin, or direction differ beyond tolerance, the mask is resampled to the image space.

5. Feature Extraction (PyRadiomics): Extracts IBSI-compliant features (excluding diagnostic features) based on the user's execution parameters.

## Execution
python /path/to/script/radiomics_extraction.py \
  --input_csv /complete/path/to/data.csv \
  --extraction_mode both \
  --mask_label 1 \
  --plane 2


  Arguments:

  - --input_csv (Required): Complete absolute path to the input CSV file. The output directory will be created here.

  - --extraction_mode: The target for feature extraction. Options: mask (Default), full, or both.

  - --mask_label: Specifies the integer label in the mask from which to extract features (Default: 1).

  - --plane: Forces 2D extraction (essential for clinical MRI). Options: 0 (Axial), 1 (Coronal), 2 (Sagittal).
## Outputs
After a successful extraction run, the following structure will be created inside the directory of your original input CSV:

/input_csv_directory/ 
└── radiomic_results/ 
    ├── features_T2_mask_1.csv 
    ├── features_T2_full_1.csv 
    └── ...

