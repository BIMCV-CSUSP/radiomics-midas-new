import SimpleITK as sitk
import os
import pandas as pd


# --- CONFIGURACIÓN DE RUTAS ---
CSV_PATH = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/updated_patients_mask.csv" 
OUTPUT_ROOT = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/crops_individuales2/"
CSV_OUTPUT_DISCS = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/updated_patients_per_discs.csv"

# Mapeo de IDs a nombres de columnas y discos
DISC_MAP = {
    1: "L5-S1",
    2: "L4-L5",
    3: "L3-L4",
    4: "L2-L3",
    5: "L1-L2"
}

def align_mask_to_image(mask, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) 
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)

def crop_and_save_discs(row, output_dir):##cambiar padding
    saved_discs_info = []
    image_path = row['T2']
    mask_path = row['mask']
    patient_id = row['patient_id']

    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = align_mask_to_image(mask, image)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)

        img_size = image.GetSize() # (Width, Height, Depth)

        os.makedirs(output_dir, exist_ok=True)

        for label_id, disc_name in DISC_MAP.items():
            if stats.HasLabel(label_id):
                bbox = stats.GetBoundingBox(label_id)
                start, size = list(bbox[:3]), list(bbox[3:])
                img_size = image.GetSize()

                # for i in range(3):
                #     orig_start = start[i]
                #     start[i] = max(0, orig_start - padding[i])
                #     desired_end = min(img_size[i], orig_start + size[i] + padding[i])
                #     size[i] = desired_end - start[i]

                # --- ADAPTIVE PADDING LOGIC ---
                new_start = []
                new_size = []
                for i in range(3):
                    if i == 2: # Z-Direction (Lateral Slices)
                        pad_px = 1 # Only 1 slice extra on each side
                    else: # X and Y (Anatomy context)
                        pad_px = int(size[i] * 0.15) # 15% context
                    
                    actual_start = max(0, start[i] - pad_px)
                    actual_end = min(img_size[i], (start[i] + size[i]) + pad_px)
                    
                    new_start.append(actual_start)
                    new_size.append(actual_end - actual_start)

                cropped_image = sitk.RegionOfInterest(image, size=new_size, index=new_start)
                out_name = os.path.join(output_dir, f"{patient_id}_{disc_name}.nii.gz")
                sitk.WriteImage(cropped_image, out_name)

                # --- NUEVA LÓGICA DE COLUMNAS ---
                # Creamos el diccionario base con la info fija
                disc_info = {
                    'patient_id': row['patient_id'],
                    'study_id': row['study_id'],
                    'T2': row['T2'],
                    'T1': row['T1'],
                    'mask': row['mask'],
                    'XNAT_name': row['XNAT_name'],
                    'disc': disc_name,
                    'disc_path': out_name,
                    # Extraemos el valor específico (el grado) de la columna correspondiente
                    'Pfirrmann': row[disc_name] if disc_name in row else None
                }
                saved_discs_info.append(disc_info)

    except Exception as e:
        print(f"Error en {patient_id}: {e}")
    
    return saved_discs_info

def run_csv_process():
    if not os.path.exists(CSV_PATH):
        print(f"Error: No existe {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    all_discs_data = []

    print(f"Procesando {len(df)} pacientes...")

    for _, row in df.iterrows():
        patient_id = row['patient_id']
        patient_out_dir = os.path.join(OUTPUT_ROOT, str(patient_id))
        if os.path.exists(str(row['T2'])) and os.path.exists(str(row['mask'])):
            discs_info = crop_and_save_discs(row, patient_out_dir)
            all_discs_data.extend(discs_info)
            print(f"--> {patient_id}: {len(discs_info)} discos procesados.")
        else:
            print(f"--> {patient_id}: Archivos no encontrados.")

    # Generar el CSV final
    df_final = pd.DataFrame(all_discs_data)
    
    # Orden deseado de columnas
    cols_order = ['patient_id', 'study_id', 'T2', 'T1', 'mask', 'XNAT_name', 'disc', 'disc_path','Pfirrmann']
    df_final = df_final[cols_order]
    
    df_final.to_csv(CSV_OUTPUT_DISCS, index=False)
    print(f"\nProceso finalizado. CSV guardado en: {CSV_OUTPUT_DISCS}")

if __name__ == "__main__":
    run_csv_process()