from pathlib import Path

import pandas as pd


def build_dataframe_from_csv(
    img_relation_path: Path, labels_path: Path, radiomic_features_path: Path
) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    labels_df.dropna(inplace=True)
    labels_df.rename(
        columns={"subject_ID": "Subject_XNAT", "ID": "Session_XNAT"}, inplace=True
    )

    midas_img_relation = pd.read_csv(img_relation_path)
    midas_img_relation["Subject_MIDS"] = midas_img_relation["Image"].map(
        lambda x: x.split("/")[8]
    )
    midas_img_relation["Session_MIDS"] = midas_img_relation["Image"].map(
        lambda x: x.split("/")[9]
    )
    midas_img_relation["Subject_XNAT"] = midas_img_relation["Subject_MIDS"].map(
        lambda x: f"ceibcs_S{int(x.split('sub-S')[1])}"
    )
    midas_img_relation["Session_XNAT"] = midas_img_relation["Session_MIDS"].map(
        lambda x: f"ceibcs_E{int(x.split('ses-E')[1])}"
    )

    id_labels = labels_df.merge(midas_img_relation, on=["Subject_XNAT", "Session_XNAT"])
    id_labels.rename(
        columns={
            "L5-S": "1",
            "L4-L5": "2",
            "L3-L4": "3",
            "L2-L3": "4",
            "L1-L2": "5",
        },
        inplace=True,
    )

    radiomic_features = pd.read_csv(radiomic_features_path, sep=",")
    radiomic_features.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

    return id_labels.merge(radiomic_features, on="ID")


def get_labels_and_features(
    img_relation_path: Path,
    labels_path: Path,
    radiomic_features_path: Path,
    label: int = 1,
) -> tuple:
    """
    Reads a CSV file from the given label and returns the labels and features as separate dataframes.

    :return: A tuple containing the labels and features.
    :rtype: tuple
    """

    data = build_dataframe_from_csv(
        img_relation_path, labels_path, radiomic_features_path
    )

    data = data.rename(columns={str(label): f"label{label}", "ID": f"label{label}ID"})
    columns_mask = data.columns.str.contains(
        f"label{label}"
    ) & ~data.columns.str.contains("Configuration")
    data = data.loc[:, columns_mask]
    data = data.rename(columns={f"label{label}": "label", f"label{label}ID": "ID"})

    label_data = data.dropna(axis=0, how="any")
    label_data = label_data.loc[label_data["label"] != 0]
    label_data["ID"] = label_data["ID"].map(lambda x: x + str(label))
    label_data = label_data.set_index("ID")
    labels = label_data["label"]
    features = label_data[
        label_data.select_dtypes(include="number").columns.tolist()
    ].drop(columns="label")
    return labels, features


def get_labels_and_features_all_discs(
    img_relation_path: Path,
    labels_path: Path,
    radiomic_features_path: Path,
    verbose: bool = False,
) -> tuple:
    """
    Get labels and features for all discs.

    :return: A tuple containing the labels and features.
    :rtype: tuple
    """
    features = []
    labels = []
    for label in range(1, 6):
        labels_i, features_i = get_labels_and_features(
            img_relation_path, labels_path, radiomic_features_path, label=label
        )
        labels.append(labels_i)
        features_i = features_i.rename(
            columns={
                name: name.replace(f"label{label}_", "")
                for name in features_i.columns.to_list()
            }
        )
        features.append(features_i)
    features = pd.concat(features, axis=0)
    labels = pd.concat(labels, axis=0)
    if verbose:
        print(f"Labels shape: {labels.shape}, Features shape: {features.shape}")
    return labels, features
