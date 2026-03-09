"""
Script for the parallel extraction of radiomic features from MRI images.

This script extracts radiomic features from multiple MRI modalities (such as T2, T1, T1C,
ADC, DWI) using PyRadiomics. It processes both the segmented mask and 
the full image. 

The script automatically detects available modalities from the input CSV columns, 
edits the YAML configuration for each modality to set dynamic parameters like bin width and spacing, 
and implements patient-level parallel processing using a ProcessPoolExecutor 
to optimize extraction time across multiple CPU cores.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import yaml


## GLOBAL CACHE (per process)
EXTRACTORS = {}

## Logging configuration
logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


##############################################################################
#                            UTILITY FUNCTIONS                               #
##############################################################################

def resample_to_reference(moving_image, reference_image, is_mask=False):
    """
    Resamples an image to the space of another reference image.
    
    Args:
        moving_image (sitk.Image): Image to be resampled
        reference_image (sitk.Image): Reference image
        is_mask (bool): If True, uses nearest neighbor interpolation to preserve label values.
                        If False, uses linear interpolation for intensity images.
    
    Returns:
        sitk.Image: Image resampled to the reference space
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(moving_image)


def bias_field_correction(image_float32, shrink_factor=4, control_points=[4, 4, 4]):
    """
    Applies N4 bias field correction to an image.
    
    Args:
        image_float32 (sitk.Image): Image in float32 format.
        shrink_factor (int): Reduction factor to accelerate processing.
        control_points (list): Control points for the N4 algorithm.
    
    Returns:
        sitk.Image: Bias field corrected image.
    """
    # Downsample the image to speed up processing
    shrinked_image = sitk.Shrink(image_float32, [shrink_factor] * image_float32.GetDimension())
    
    # Configure and apply the N4 filter
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetNumberOfControlPoints(control_points)
    bias_field_filter.UseMaskLabelOff()
    
    bias_field_filter.Execute(shrinked_image)
    
    # Apply the correction to the original image
    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image_float32)
    bias_corrected_image = image_float32 / sitk.Exp(log_bias_field)
    
    return bias_corrected_image


# def preprocess_image(image):
#     """
#     Applies preprocessing to an image: float32 conversion, bias field correction, and noise reduction.
    
#     Args:
#         image (sitk.Image): Original image.
    
#     Returns:
#         sitk.Image: Preprocessed image.
#     """
#     # Convert to float32 for numerical operations
#     image_float32 = sitk.Cast(image, sitk.sitkFloat32)

#     # Bias field correction to normalize intensities
#     bias_corrected_image = bias_field_correction(image_float32)

#     # Noise reduction using Curvature Anisotropic Diffusion
#     denoised_image = sitk.CurvatureAnisotropicDiffusion(bias_corrected_image, timeStep=0.01)

#     return denoised_image

def preprocess_image(image, time_step):
    """
    Preprocesa la imagen usando el time_step precalculado para la modalidad.
    """
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)
    bias_corrected_image = bias_field_correction(image_float32)

    # Aplicamos el filtro con el valor fijo recibido
    denoised_image = sitk.CurvatureAnisotropicDiffusion(
        bias_corrected_image, 
        timeStep=time_step,
        numberOfIterations=5,
        conductanceParameter=1.0
    )

    return denoised_image


def calculate_optimal_bin_width(df, modality, nb_bins=64):
    """
    Calcula el binWidth promedio para una modalidad específica basándose en el rango 
    de intensidades (Max - Min) de las imágenes.
    """
    logger.info(f"Calculando binWidth óptimo para {modality}...")
    ranges = []
    # Usamos una muestra de hasta 20 imágenes para no ralentizar demasiado el inicio
    sample_size = min(len(df), 20)
    sample_df = df.sample(sample_size)

    for _, row in sample_df.iterrows():
        try:
            img_path = row[modality]
            if not os.path.exists(img_path): continue
            img = sitk.ReadImage(img_path)
            img = sitk.DICOMOrient(img, 'RAS')
            statistics = sitk.StatisticsImageFilter()
            statistics.Execute(img)
            img_range = statistics.GetMaximum() - statistics.GetMinimum()
            ranges.append(img_range)
        except:
            continue

    if not ranges:
        return None
    
    avg_range = np.mean(ranges)
    optimal_bw = avg_range / nb_bins
    logger.info(f"Modality {modality}: Range Medio={avg_range:.2f}, binWidth sugerido={optimal_bw:.4f}")
    return optimal_bw


def calculate_target_spacing(df, modality, plane=None):
    """
    Calcula el espaciado (spacing) promedio de los voxels para una modalidad.
    """
    logger.info(f"Calculando spacing objetivo para {modality}...")
    spacings = []
    
    # Tomamos una muestra para eficiencia, o df['Image'] completo
    sample_df = df.dropna(subset=[modality])
    
    for _, row in sample_df.iterrows():
        try:
            img_path = row[modality]
            img = sitk.ReadImage(img_path)
            img_ras = sitk.DICOMOrient(img, 'RAS')
            # Ahora GetSpacing() devuelve los valores en orden RAS (X, Y, Z)
            spacings.append(img_ras.GetSpacing())
        except Exception as e:
            continue

    if not spacings:
        return None
    
    # Media por eje
    avg_spacing = np.mean(spacings, axis=0)
    final_spacing = [float(avg_spacing[0]), float(avg_spacing[1]), float(avg_spacing[2])]

    # # El componente perpendicular al plano seleccionado debe ser 0 
    # # para evitar interpolación o conectividad entre cortes.
    # if plane is not None:
    #     logger.info(f"Modo 2D detectado (Plano {plane}): Anulando eje perpendicular.")
    #     if plane == 0: final_spacing[2] = 0.0  # Axial (Z)
    #     elif plane == 1: final_spacing[1] = 0.0 # Coronal (Y)
    #     elif plane == 2: final_spacing[0] = 0.0 # Sagital (X)
    # else:
    #     logger.info(f"Modo 3D detectado para {modality}: Manteniendo spacing volumétrico.")
        
    return final_spacing


def get_extractor(modality_key, label_value=None, plane=None, custom_bin_width=None, resampled_spacing=None):
    """
    Returns an initialized RadiomicsFeatureExtractor for the specified modality.
    The extractor is created once per process and then reused (caching).
    It searches for the parameters .yaml file in the same directory as this script.
    
    Args:
        modality_key (str): The imaging modality (e.g., 'T2', 'ADC').
        label_value (int, optional): Specific label ID to extract features from.
    """

    # Get the directory path where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, f"Params_template.yaml")
    
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(script_dir), "radiomics_template", "Params_template.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Not found: {yaml_path}")

    is_adc = modality_key.upper() == "ADC"
    
    # CLAVE DE CACHÉ ROBUSTA: Incluye todos los parámetros variables
    # Convertimos el spacing a tupla para que sea hashable
    spacing_tuple = tuple(resampled_spacing) if resampled_spacing is not None else None
    cache_key = (modality_key.upper(), label_value, plane, custom_bin_width, spacing_tuple, is_adc)

    if cache_key not in EXTRACTORS:
        logger.info(f"Instanciando nuevo extractor para: {modality_key} (ADC={is_adc})")
        extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)

        # Sobreescritura de parámetros dinámicos
        settings_update = {}
        if label_value is not None: settings_update["label"] = int(label_value)

        if plane is not None:
            settings_update["force2D"] = True
            settings_update["force2Ddimension"] = int(plane)
        else:
            settings_update["force2D"] = False
            logger.info(f"Extracción en modo 3D para {modality_key}.")

        if custom_bin_width is not None: settings_update["binWidth"] = float(custom_bin_width)

        if resampled_spacing is not None: settings_update["resampledPixelSpacing"] = resampled_spacing
            
        # Regla específica para ADC: No normalizar para mantener escala física
        if is_adc:
            settings_update["normalize"] = False
            logger.info("ADC detectado: Normalización desactivada para esta instancia.")
        
        extractor.settings.update(settings_update)
        EXTRACTORS[cache_key] = extractor

    return EXTRACTORS[cache_key]


def create_full_mask(reference_image):
    """
    Creates a dummy mask covering the entire extent of the reference image.
    This is used for whole-image radiomic extraction when no specific 
    segmentation is required.
    
    Args:
        reference_image (sitk.Image): The image to use as a template for size and spatial metadata.
    
    Returns:
        sitk.Image: A binary mask where all voxels are set to 1.
    """
    # Initialize a blank image with the same size and unsigned 8-bit integer type
    mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
    # Copy spatial information (Origin, Spacing, Direction) to ensure alignment
    mask.CopyInformation(reference_image)
    # Set all voxels to 1 to create a global mask (background is usually 0)
    mask = mask + 1

    return mask


def check_geometry_consistency(image, mask, tolerance=1e-5):
    """
    Checks if the image and mask are spatially aligned within a specific tolerance.
    Verifies dimensions, spacing, origin, and direction matrix.
    
    Args:
        image (sitk.Image): The intensity image.
        mask (sitk.Image): The segmentation mask.
        tolerance (float): Maximum allowed difference for spatial metadata.
    
    Returns:
        bool: True if the geometry is consistent, False otherwise.
    """
    # Verify that both have the same number of dimensions
    if image.GetDimension() != mask.GetDimension():
        return False
    # Calculate the maximum absolute difference for Spacing, Origin, and Direction
    spacing_diff = np.max(np.abs(np.array(image.GetSpacing()) - np.array(mask.GetSpacing())))
    origin_diff = np.max(np.abs(np.array(image.GetOrigin()) - np.array(mask.GetOrigin())))
    direction_diff = np.max(np.abs(np.array(image.GetDirection()) - np.array(mask.GetDirection())))

    # Return True only if all differences are within the defined tolerance
    return (spacing_diff <= tolerance and origin_diff <= tolerance and direction_diff <= tolerance)


def get_label_from_extractor(extractor):
    """
    Extracts the 'label' parameter defined in the extractor's YAML configuration.
    
    Args:
        extractor (featureextractor.RadiomicsFeatureExtractor): The initialized extractor.
    
    Returns:
        int: The label value. Defaults to 1 if not explicitly defined (PyRadiomics default).
    """
    # Access the internal settings dictionary of the PyRadiomics extractor
    return extractor.settings.get("label", 1)


def save_resultant_yaml(modality, base_yaml_path, output_dir, final_settings):
    """Guarda el archivo YAML final convirtiendo tipos NumPy a nativos de Python."""
    try:
        with open(base_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'setting' not in config: config['setting'] = {}
        
        # --- LIMPIEZA DE TIPOS NUMPY ---
        clean_settings = {}
        for k, v in final_settings.items():
            if isinstance(v, (np.float32, np.float64, np.ndarray)):
                if isinstance(v, np.ndarray) or isinstance(v, list):
                    clean_settings[k] = [float(i) for i in v]
                else:
                    clean_settings[k] = float(v)
            else:
                clean_settings[k] = v

        # Actualizar configuración
        config['setting'].update(clean_settings)
        
        # REGLA ADC: No normalizar
        if modality.upper() == "ADC":
            logger.info(f"Modalidad ADC: Forzando normalize a False.")
            config['setting']['normalize'] = False
        
        out_path = os.path.join(output_dir, f"Params_{modality}.yaml")
        with open(out_path, 'w') as f:
            # default_flow_style=False hace que las listas se vean como guiones abajo
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"YAML final generado con éxito: {out_path}")
    except Exception as e:
        logger.error(f"Error guardando YAML para {modality}: {e}")


def extract_radiomic_features(extractor, image, mask, patient_id, study_id):
    """
    Executes the radiomic feature extraction and filters the results.
    
    Args:
        extractor (featureextractor.RadiomicsFeatureExtractor): The configured extractor.
        image (sitk.Image): The preprocessed intensity image.
        mask (sitk.Image): The segmentation mask (ROI).
        patient_id (str): Unique identifier for the patient.
        study_id (str): Unique identifier for the study/session.
    
    Returns:
        dict: A dictionary containing patient metadata and the extracted radiomic features.
    """

    
    # --- DIAGNOSE ---
    mask = sitk.Cast(mask, sitk.sitkUInt8)  
    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(mask)
    label_id = int(extractor.settings.get('label', 1))
    
    if lssif.HasLabel(label_id):
        num_voxels = lssif.GetNumberOfPixels(label_id)
        logger.info(f">>> [{patient_id}] Modality: OK | Voxels: {num_voxels}")
    else:
        logger.warning(f">>> [{patient_id}] Modality: FALLO | La etiqueta {label_id} NO EXISTE en esta máscara.")
        return None
    
    # Run the PyRadiomics extraction engine
    features = extractor.execute(image, mask)

    # Initialize the output dictionary with identifiers
    out = {"patient_id": patient_id, "study_id": study_id}

    # Filter out diagnostic information to keep only the actual radiomic features
    for k, v in features.items():
        if not k.startswith("diagnostics"):
            out[k] = v
    return out


# ==============================
# PATIENT-LEVEL PROCESSING
# ==============================

def process_patient(row, extraction_mode, modalities,label_value, dynamic_settings, plane):
    """
    Handles the complete extraction workflow for a single patient across multiple modalities.
    Includes image loading, preprocessing, spatial alignment, and feature extraction.
    
    Args:
        row (dict): A dictionary representing a CSV row with file paths and IDs.
        extraction_mode (str): Mode of extraction ('mask', 'full', or 'both').
        modalities (list): List of imaging modalities to process (e.g., ['T2', 'ADC']).
        label_value (int): Specific label ID for the ROI extraction.
    
    Returns:
        dict: Nested dictionary containing features organized by modality and mask type.
              Returns None if the base mask is missing or unreadable.
    """
    patient_id = row["patient_id"]
    study_id = row["study_id"]
    mask_path = row["mask"]

    # Check if the segmentation mask file exists
    if not os.path.isfile(mask_path):
        logger.warning(f"Mask not found for patient {patient_id}")
        return None

    try:
        mask_image = sitk.ReadImage(mask_path)
        mask_image = sitk.DICOMOrient(mask_image, 'RAS')
    except Exception as e:
        logger.error(f"Error reading mask for patient {patient_id}: {e}")
        return None

    results = {}

    for modality in modalities:
        image_path = row.get(modality)
        # Skip if the modality path is missing or the file doesn't exist
        if not image_path or not os.path.isfile(image_path):
            logger.warning(f"Image for modality {modality} not found for patient {patient_id}")
            continue

        try:
            # EXTRAER PARÁMETROS DINÁMICOS
            bw = dynamic_settings[modality]["binWidth"]
            sp = dynamic_settings[modality]["spacing"]

            # Initialize/retrieve the extractor and preprocess the image
            extractor = get_extractor(modality, label_value, plane, custom_bin_width=bw, resampled_spacing=sp)
            
            image = sitk.ReadImage(image_path)
            image = sitk.DICOMOrient(image, 'RAS')
            image = preprocess_image(image, time_step=dynamic_settings[modality]["time_step"])

            # Spatial consistency check: Resample mask if it doesn't match image geometry
            if not check_geometry_consistency(image, mask_image):
                logger.warning(f"Resampling mask for patient {patient_id} modality {modality}")
                mask_resampled = resample_to_reference(mask_image, image, is_mask=True)
            else:
                mask_resampled = mask_image

            results[modality] = {}

            # Extraction for the specific ROI (segmented mask)
            if extraction_mode in ("mask", "both"):
                feats_mask = extract_radiomic_features(extractor, image, mask_resampled, patient_id, study_id)
                results[modality]["mask"] = feats_mask

            # Extraction for the whole image (full mask)
            if extraction_mode in ("full", "both"):
                # full_mask = create_full_mask(image)
                # print(full_mask)
                # feats_full = extract_radiomic_features(extractor, image, full_mask, patient_id, study_id)
                # results[modality]["full"] = feats_full
                # NUEVA LÓGICA: Generar la máscara 'full' con el tamaño resampleado
                # Obtenemos el tamaño pre-calculado en el main
                resampled_size = dynamic_settings[modality]["resampled_size"]
                
                # Crear la máscara full directamente con ese tamaño
                full_mask = sitk.Image(resampled_size, sitk.sitkUInt8)
                
                safe_spacing = [
                    ns if ns > 0 else orig_s 
                    for ns, orig_s in zip(sp, image.GetSpacing())
                ]
                
                # Copiar la información espacial de la imagen original (PyRadiomics la ajustará)
                full_mask.SetOrigin(image.GetOrigin())
                full_mask.SetDirection(image.GetDirection())
                # IMPORTANTE: El spacing debe ser el objetivo (sp)
                full_mask.SetSpacing(safe_spacing) 
                
                # Rellenar con Label 1
                full_mask = full_mask + 1
                
                feats_full = extract_radiomic_features(extractor, image, full_mask, patient_id, study_id)
                results[modality]["full"] = feats_full


        except Exception as e:
            logger.error(f"Error in {modality} for patient {patient_id}: {e}", exc_info=True)

    return results



# ==============================
# MAIN
# ==============================

def main(input_csv, extraction_mode, label_value, plane):
    """
    Main execution pipeline: loads data, orchestrates parallel processing, 
    and saves the extracted features to CSV files.
    
    Args:
        input_csv (str): Path to the CSV file containing patient IDs and file paths.
        extraction_mode (str): Extraction target ('mask', 'full', or 'both').
        label_value (int): Specific label ID to override YAML settings.
    """
    
    df = pd.read_csv(input_csv)

    # Set up the output directory relative to the input file
    output_dir = os.path.join(os.path.dirname(input_csv), "radiomic_results")
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_yaml = os.path.join(os.path.dirname(script_dir), "radiomics_template/Params_template.yaml")

    # Validate input: Ensure essential columns exist in the CSV
    required_cols = ["patient_id", "mask"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return

    # Definimos las que buscamos
    target_modalities = ["T1", "T1C", "T2", "ADC", "DWI"]
    modalities = []

    print("----", label_value)
    for m in target_modalities:
        # Buscamos si existe la columna (sin importar mayúsculas) y si tiene al menos un dato no nulo
        found_col = [col for col in df.columns if col.upper() == m]
        if found_col:
            actual_col = found_col[0]
            if df[actual_col].notna().any():
                modalities.append(actual_col)
    
    if not modalities:
        logger.error(f"No se detectó ninguna de las columnas {target_modalities} en el CSV.")
        return

    logger.info(f"Modalidades detectadas para procesar: {modalities}")


    # Diccionarios para almacenar los parámetros calculados
    dynamic_settings = {m: {"binWidth": None, "spacing": None} for m in modalities}

    for m in modalities:
        # 1. Calcular Bin Width
        bw = calculate_optimal_bin_width(df, m)
        # 2. Calcular Spacing
        sp = calculate_target_spacing(df, m, plane)
        
        # Límite de estabilidad para 3D (divisor 10 para seguridad extra)
        min_sp = min(sp)
        ts = (min_sp**2) / 10.0
        modality_time_step = min(ts, 0.01)
        
        # Guardarlo en dynamic_settings para que process_patient lo reciba
        dynamic_settings[m] = {
            "spacing": sp,
            "binWidth": bw,
            "time_step": modality_time_step  # <--- Lo guardamos aquí
        }
        
        logger.info(f"Modalidad {m}: Spacing={sp}, BinWidth={bw}, TimeStep={modality_time_step:.6f}")

        # ------
        # NUEVO: Pre-calcular geometría resampleada para el modo 'full'
        # Usamos una imagen de muestra para obtener la geometría original y proyectar la nueva
        sample_img_path = df[m].dropna().iloc[0]
        sample_img = sitk.ReadImage(sample_img_path)
        
        # Calcular tamaño de salida basado en el spacing objetivo (sp)
        original_size = sample_img.GetSize()
        original_spacing = sample_img.GetSpacing()
        new_size = []
        for sz, ors, ns in zip(original_size, original_spacing, sp):
            if ns > 0:
                new_size.append(int(sz * ors / ns))
            else:
        # Si el spacing es 0 (eje anulado), mantenemos el tamaño original 
        # para que la máscara tenga al menos una capa.
                new_size.append(sz)
        # new_size = [int(sz * os / ns) for sz, os, ns in zip(original_size, original_spacing, sp)]
        
        dynamic_settings[m]["resampled_size"] = new_size # Guardamos el tamaño

        # ----
        # Guardar el nuevo archivo YAML
        final_params = {}
        if bw: final_params["binWidth"] = bw
        if sp: final_params["resampledPixelSpacing"] = sp
        if label_value: final_params["label"] = label_value
        if plane: final_params["force2Ddimension"] = plane
        
        save_resultant_yaml(m, base_yaml, output_dir, final_params)

    # Parallel processing setup
    max_workers = min(4, multiprocessing.cpu_count())
    results = {m: {"mask": [], "full": []} for m in modalities}
    
    # Execute patient-level processing in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_patient, row.to_dict(), extraction_mode, modalities, label_value, dynamic_settings, plane)
            for _, row in df.iterrows()
        ]


        # Monitor progress and collect results as they finish
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patients"):
            patient_data = future.result()
            if patient_data is None:
                continue
            # Merge parallel results into the main results dictionary
            print(patient_data.keys())

            for modality in patient_data:
                if modality in results:
                    if "mask" in patient_data[modality]:
                        results[modality]["mask"].append(patient_data[modality]["mask"])
                    if "full" in patient_data[modality]:
                        results[modality]["full"].append(patient_data[modality]["full"])


    # # Final saving process: Create individual CSVs per modality and extraction type
    # for modality in modalities:
    #     bw = dynamic_settings[modality]["binWidth"]
    #     sp = dynamic_settings[modality]["spacing"]
        
    #     # Coincidir con los parámetros usados en el procesamiento
    #     extractor_instance = get_extractor(modality, label_value, plane, custom_bin_width=bw, resampled_spacing=sp)
    #     current_label = get_label_from_extractor(extractor_instance)

    #     # Save ROI mask features
    #     if extraction_mode in ("mask", "both") and results[modality]["mask"]:
    #         out_mask = os.path.join(output_dir, f"features_{modality}_mask_{current_label}.csv")
    #         pd.DataFrame(results[modality]["mask"]).to_csv(out_mask, index=False)
    #         logger.info(f"Saved {out_mask}")
        
    #     # Save whole-image features
    #     if extraction_mode in ("full", "both") and results[modality]["full"]:
    #         out_full = os.path.join(output_dir, f"features_{modality}_full_{current_label}.csv")
    #         pd.DataFrame(results[modality]["full"]).to_csv(out_full, index=False)
    #         logger.info(f"Saved {out_full}")

    # logger.info("Extraction completed")
    # Final saving process: Create individual CSVs per modality and extraction type
    for modality in modalities:
        bw = dynamic_settings[modality]["binWidth"]
        sp = dynamic_settings[modality]["spacing"]
        
        extractor_instance = get_extractor(modality, label_value, plane, custom_bin_width=bw, resampled_spacing=sp)
        current_label = get_label_from_extractor(extractor_instance)

        # --- GUARDADO PARA MÁSCARA (ROI) ---
        if extraction_mode in ("mask", "both") and results[modality]["mask"]:
            out_mask = os.path.join(output_dir, f"features_{modality}_mask_{current_label}.csv")
            failed_indices = [i for i, res in enumerate(results[modality]["mask"]) if res is None]
            # FILTRO CRÍTICO: Solo tomamos elementos que no sean None
            clean_mask_data = [res for res in results[modality]["mask"] if res is not None]
            
            if clean_mask_data:
                pd.DataFrame(clean_mask_data).to_csv(out_mask, index=False)
                logger.info(f"Saved {out_mask} ({len(clean_mask_data)} patients)")

                if failed_indices:
                    logger.warning(f"ATENCIÓN: {len(failed_indices)} pacientes fallaron en {modality}.")
                    
                    # CAMBIO CLAVE: Usamos la primera columna del CSV original como ID 
                    # para evitar el KeyError si 'PatientID' no existe.
                    id_col_name = df.columns[0]
                    failed_ids = df.iloc[failed_indices][id_col_name].tolist()
                    
                    print(f"\n[!] Los siguientes pacientes NO se incluyeron en el CSV ({modality}):")
                    for fid in failed_ids:
                        print(f"    - {fid}")
                    print("")

            else:
                logger.error(f"No valid mask results for {modality}")

        # --- GUARDADO PARA IMAGEN COMPLETA (FULL) ---
        if extraction_mode in ("full", "both") and results[modality]["full"]:
            out_full = os.path.join(output_dir, f"features_{modality}_full_{current_label}.csv")
            
            # FILTRO CRÍTICO: Solo tomamos elementos que no sean None
            clean_full_data = [res for res in results[modality]["full"] if res is not None]
            
            if clean_full_data:
                pd.DataFrame(clean_full_data).to_csv(out_full, index=False)
                logger.info(f"Saved {out_full} ({len(clean_full_data)} patients)")
            else:
                logger.error(f"No valid full-image results for {modality}")

    logger.info("Extraction completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--extraction_mode", default="mask", choices=["mask", "full", "both"])
    parser.add_argument("--plane", type=int, default=None, help="axial=0, coronal=1, sagittal=2")
    parser.add_argument("--mask_label", type=int, default=1, help="Override the 'label' parameter in the YAML configuration for ROI extraction.")
    
    args = parser.parse_args()
    
    main(args.input_csv, args.extraction_mode, args.mask_label, args.plane)

## Example execution: 
## python radiomics_extraction.py --input_csv /path/to/data.csv --extraction_mode mask --mask_label 1 --plane 0
