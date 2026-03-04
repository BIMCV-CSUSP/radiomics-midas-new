#!/usr/bin/env python3
"""
convert_to_nifti.py

A script to convert a DICOM image directory (and optionally a DICOM segmentation file)
to NIfTI format using MONAI transforms.

Usage:
    python convert_to_nifti.py \
        --image_dir /path/to/dicom_image_folder \
        [--seg_file /path/to/dicom_segmentation.dcm] \
        --output_dir /path/to/output_folder
"""

import os
import argparse
from monai.transforms import (
    LoadImaged,
    AsDiscreted,
    ResampleToMatchd,
    SaveImaged,
)
from monai.data import ITKReader



def convert_to_nifti(image_dir: str,
                     seg_file: str,
                     output_dir: str,
                     case_name: str = None) -> None:
    """
    Convert a DICOM image directory (and optionally a DICOM segmentation file)
    to NIfTI format.

    Args:
        image_dir (str): Path to the directory containing DICOM image slices.
        seg_file (str): Path to the DICOM segmentation file (single .dcm). Pass None if no segmentation.
        output_dir (str): Directory where the converted .nii.gz files will be saved.
        case_name (str): Base name to use for output filenames. If None, derived from image_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if case_name is None:
        case_name = os.path.basename(os.path.normpath(image_dir))

    # Prepare data_dict for MONAI loader
    data_dict = {"image": image_dir}
    if seg_file:
        data_dict["segmentation"] = seg_file

    # Define loader: will load both image directory and segmentation if provided
    loader = LoadImaged(
        keys=list(data_dict.keys()),
        image_only=False,
        # reader="ITKReader",
        ensure_channel_first=True,
    )

    try:
        # Load data
        loaded = loader(data_dict)

        if seg_file:
            # Binarize the segmentation (threshold = 1)
            loaded = AsDiscreted(keys="segmentation", threshold=1)(loaded)
            # Resample segmentation to match image spacing/resolution
            loaded = ResampleToMatchd(
                keys="segmentation",
                key_dst="image",
                mode="nearest",
                padding_mode="zeros",
            )(loaded)

        # Prepare SaveImaged transform
        keys_to_save = ["image"]
        if seg_file:
            keys_to_save.append("segmentation")

        saver = SaveImaged(
            keys=keys_to_save,
            output_dir=output_dir,
            output_ext=".nii.gz",
            output_postfix=case_name,
            separate_folder=False,
            squeeze_end_dims=True,
        )

        # Save the outputs
        saver(loaded)
        print(f"✅ Conversion successful for case '{case_name}'.")
        if seg_file:
            print(f"   Image NIfTI: {os.path.join(output_dir, case_name + '_image.nii.gz')}")
            print(f"   Seg NIfTI:   {os.path.join(output_dir, case_name + '_segmentation.nii.gz')}")
        else:
            print(f"   Image NIfTI: {os.path.join(output_dir, case_name + '_image.nii.gz')}")

    except Exception as e:
        print(f"❌ Error converting case '{case_name}': {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DICOM image directory (and optional DICOM segmentation) to NIfTI."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the folder containing DICOM image slices.",
    )
    parser.add_argument(
        "--seg_file",
        type=str,
        default=None,
        help="(Optional) Path to a single DICOM segmentation file (.dcm).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output NIfTI files will be saved.",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default=None,
        help="(Optional) Base name for output files. Defaults to the image_dir folder name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_nifti(
        image_dir=args.image_dir,
        seg_file=args.seg_file,
        output_dir=args.output_dir,
        case_name=args.case_name,
    )
