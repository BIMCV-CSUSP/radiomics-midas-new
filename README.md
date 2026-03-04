# Intervertebral Disc Analysis Pipeline
This repository provides an automated pipeline for lumbar intervertebral disc analysis, encompassing segmentation, preprocessing, and two distinct modeling approaches: Radiomics (Machine Learning) and Deep Learning (DenseNet).

## Pipeline Workflow
The project is structured into three main phases:

### Phase 1: Preprocessing & Segmentation
This phase isolates the intervertebral discs from raw medical images.

- Segmentation: Run segmentation.py to execute TotalSegmentator and extract the initial disc masks.

- Label Modification: Execute label_modification.py to reorder the disc masks sequentially (from bottom to top) and filter the labels to maintain only the 5 target discs (removing any labels >5).

- CSV Creation: Run creation_mask_csv.py to generate a master CSV file that maps the original image paths to the newly generated mask paths.

### Phase 2: Modeling Options
Once the data is preprocessed, you can choose between two analytical paths:

- Option 1: Machine Learning (Radiomics)
This approach extracts quantitative features from the segmented disc masks.

  - Extraction: Run radiomics_extraction.py using your CSV and mask label to generate feature matrices.

  - Training: Execute train_binary.py to perform feature selection and train classification models on the radiomic features.

- Option 2: Deep Learning (DenseNet)
This approach uses image crops to feed a neural network directly.

  - Cropping: Execute cropped.py to extract 5 individual images (one per disc) and create a new CSV mapping these crops to their respective paths.

  - Training: Run densenet_training.py --csv <path_to_csv> to train the DenseNet model on the disc crops.
