import monai.transforms as transforms
import numpy as np
import SimpleITK as sitk
from radiomics import imageoperations


class CheckMaskVol(transforms.MapTransform):
    def __init__(
        self,
        keys=["image", "mask"],
        minimum_roi_dimensions: int = 3,
        minimum_roi_size: int = 1000,
    ):
        super().__init__(keys)
        self.minimum_roi_dimensions = minimum_roi_dimensions
        self.minimum_roi_size = minimum_roi_size

    def __call__(self, x):
        image = sitk.ReadImage(x[self.keys[0]])
        mask = sitk.ReadImage(x[self.keys[1]])
        labels = np.unique(sitk.GetArrayFromImage(mask).ravel())
        valid_labels = []
        for label in labels:
            if label != 0:
                try:
                    imageoperations.checkMask(
                        image,
                        mask,
                        minimumROIDimensions=self.minimum_roi_dimensions,
                        minimumROISize=self.minimum_roi_size,
                        label=label,
                    )
                    result = label
                except Exception as e:
                    result = None
                if result:
                    valid_labels.append(result)
        x["valid_labels"] = valid_labels[:5]
        return x


class CropForegroundd(transforms.MapTransform):
    def __init__(
        self, keys=["image"], source_key="mask", margin=0, k_divisible=(64, 64, 1)
    ):
        super().__init__(keys)
        self.k_divisible = k_divisible
        self.margin = margin
        self.source_key = source_key

    def __call__(self, x):
        key = self.keys[0]
        input_data = {"image": x[key], "mask": x[self.source_key]}
        discs = []
        labels = []
        for label, disc in enumerate(x["valid_labels"], start=1):
            select_fn = lambda x: x == disc
            crop = transforms.CropForegroundd(
                keys=self.keys,
                source_key=self.source_key,
                select_fn=select_fn,
                margin=self.margin,
                k_divisible=self.k_divisible,
            )(input_data)
            discs.append(crop["image"])
            labels.append(x[str(label)])

        return [{"image": disc, "label": label} for disc, label in zip(discs, labels)]
