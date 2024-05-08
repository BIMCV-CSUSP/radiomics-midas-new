# author: Adolfo López Cerdán
# email: adlpecer@gmail.com
# Description: Radiomics features extraction with Pyradiomics using multiprocessing
# and pandas formatting. Strongly based in Pyradiomics examples (http://radiomics.io).
# Adapted by: Hector Carceller, Joaquim Montell and Jesus Alejandro Alzate
# Adapted by: Elena Oliver-Garcia 20230522
# Adapted by: Alejandro Mora-Rubio 20240125

from __future__ import print_function

import logging
import threading
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations


# Perturbation
SHIFT_MASK = False
EROSION = False
DILATION = True
# Source Image
CENTER_SLICE = False


def checkMaskVol(image, mask, label):
    try:
        imageoperations.checkMask(
            image, mask, minimumROIDimensions=3, minimumROISize=1000, label=label
        )
        result = label
    except Exception as e:
        result = None
    return result


def run(row):
    start_time = time.time()

    _, case = row
    logger = rLogger.getChild(case["ID"])
    threading.current_thread().name = case["ID"]

    if params.exists():
        extractor = featureextractor.RadiomicsFeatureExtractor(str(params))
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings["binWidth"] = 25
        settings["resampledPixelSpacing"] = None
        settings["interpolator"] = sitk.sitkBSpline
        settings["enableCExtensions"] = True
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    logger.info(
        "Processing Patient %s (Image: %s, Mask: %s)",
        case["ID"],
        case["Image"],
        case["Mask"],
    )

    image_path = case["Image"]
    mask_path = case["Mask"]
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    labels = np.unique(sitk.GetArrayFromImage(mask).ravel())
    valid_labels = []
    for label in labels:
        result = checkMaskVol(image, mask, label)
        if result:
            valid_labels.append(result)

    if CENTER_SLICE:
        center_slice = image.GetSize()[2] // 2
        image = image[:, :, center_slice]
        mask = mask[:, :, center_slice]

    if SHIFT_MASK:
        max_shift_x = 3
        max_shift_y = 5
        dx = np.random.randint(-max_shift_x, max_shift_x + 1)
        dy = np.random.randint(-max_shift_y, max_shift_y + 1)
        translation = sitk.TranslationTransform(3, [dx, dy, 0])
        mask = sitk.Resample(mask, translation)

    if EROSION:
        for i in np.array(valid_labels[:5]).astype(np.double):
            mask = sitk.BinaryErode(mask, foregroundValue=i)

    if DILATION:
        for i in np.array(valid_labels[:5]).astype(np.double):
            mask = sitk.BinaryDilate(mask, foregroundValue=i)

    patient = []
    for index, label in enumerate(valid_labels[:5], start=1):
        label = int(label)
        logger.info(
            "Processing Patient %s (Image: %s, Mask: %s, Label: %s)",
            case["ID"],
            case["Image"],
            case["Mask"],
            label,
        )
        if (image_path is not None) and (mask_path is not None):
            try:
                result = pd.Series(extractor.execute(image, mask, label))
            except Exception:
                logger.error("FEATURE EXTRACTION FAILED:", exc_info=True)
                result = pd.Series()
        else:
            logger.error("FEATURE EXTRACTION FAILED: Missing Image and/or Mask")
            result = pd.Series()

        result.name = case["ID"]
        result = result.add_prefix("label{}_".format(index))
        patient.append(result)
    if len(patient) == 0:
        logger.error(f"FEATURE EXTRACTION FAILED: {case['ID']}")
        patient = pd.Series()
        patient.name = case["ID"]
    elif len(patient) == 1:
        patient = patient[0]
    else:
        patient = pd.concat(patient, axis=0)

    execution_time = time.time() - start_time

    return patient, execution_time


def main():
    # Initialize logging for batch log messages
    logger = rLogger.getChild("batch")
    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)
    logger.info("pyradiomics version: %s", radiomics.__version__)
    logger.info("Loading CSV")

    try:
        # Use pandas to read and transpose ('.T') the input data
        # The transposition is needed so that each column represents one test case
        # This is easier for iteration over the input cases.
        df = pd.read_csv(inputCSV)
    except Exception:
        logger.error("CSV READ FAILED", exc_info=True)
        exit(-1)
    logger.info("Loading Done")
    logger.info("Patients: %d", df.shape[0])

    pool = Pool(processes=(cpu_count() - 1))
    pool_results = pool.map(run, df.iterrows())
    l_results = [result[0] for result in pool_results]
    l_times = [result[1] for result in pool_results]
    pool.close()
    pool.join()

    # Merge results in one df
    results = pd.DataFrame()
    for result in l_results:
        results = results.join(result, how="outer")
    # general info and features in two files
    # Creating an only-features CSV makes R loading easier
    results = results.T
    logger.info("Extraction complete, writing CSVs")
    results.to_csv(outputFilepath, na_rep="NaN")
    logger.info("Features CSV writing complete")
    logger.info(
        f"Average execution time: {np.mean(l_times):.2f} +/- {np.std(l_times):.2f} seconds"
    )


if __name__ == "__main__":

    import argparse

    threading.current_thread().name = "Main"
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    homedir = Path(__file__)

    parser = argparse.ArgumentParser(
        prog="Radiomics Feature Extractor for MIDAS",
        description="This program takes a CSV with paths to MRI nifti files, and their corresponding segmentation masks,"
        " and computes the radiomic features specified in the parameter file.",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="the path to the input CSV",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="the path to the output CSV",
    )
    parser.add_argument(
        "--param",
        type=Path,
        default=homedir / "Params.yaml",
        help="the path to the pyradiomics parameter file (default: `src/Params.yaml`)",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="the path to the log file (default: None, log to console)",
    )
    args = parser.parse_args()

    inputCSV = args.input_csv
    outputFilepath = args.output_csv
    params = args.param

    # Configure logging
    rLogger = radiomics.logger
    if progress_filename := args.logfile:
        # Create handler for writing to log file
        handler = logging.FileHandler(filename=progress_filename, mode="a")
        handler.setFormatter(
            logging.Formatter("%(levelname)s: (%(threadName)s) %(name)s: %(message)s")
        )
        rLogger.addHandler(handler)

    main()
