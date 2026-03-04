import pandas as pd
import subprocess
import os
from pathlib import Path

## source /mnt/datalake/openmind/MedP-Midas/venv/bin/activate
# 1. Configuración de rutas
csv_path = '/mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/tu_archivo_limpio.csv'  # Nombre de tu archivo

# Creamos la carpeta 'segmentations' en el mismo directorio que el CSV
base_dir = Path(csv_path).parent
output_base_dir = base_dir / "segmentations"
output_base_dir.mkdir(parents=True, exist_ok=True)
 
# 2. Cargar el CSV
# Asumo que tus columnas se llaman 'ruta_imagen' e 'id_paciente'
df = pd.read_csv(csv_path)
print(f"Total de imágenes a procesar: {len(df)}")
 
# 3. Bucle de procesamiento
for index, row in df.iterrows():

    img_path = row['T2']

    patient_id = str(row['patient_id'])

    # Creamos una subcarpeta para cada paciente dentro de 'segmentations'

    patient_output_dir = output_base_dir / patient_id

    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # Verificación de existencia (para no repetir trabajo si el proceso se corta)

    # TotalSegmentator suele generar múltiples archivos o un .nii.gz

    # Aquí chequeamos si la carpeta ya tiene contenido

    if any(patient_output_dir.iterdir()):

        print(f"Skipping {patient_id}: Ya segmentado.")

        continue
 
    print(f"[{index+1}/{len(df)}] Procesando Paciente: {patient_id}")

    # 4. Ejecución del comando

    comando = [

        "TotalSegmentator",

        "-i", img_path,

        "-o", str(patient_output_dir),

        "-ta", "total_mr",
        "--roi_subset","intervertebral_discs"

    ]

    try:

        # Usamos subprocess para capturar errores de la terminal

        subprocess.run(comando, check=True)

        print(f"✓ Éxito: {patient_id}")

    except subprocess.CalledProcessError as e:

        print(f"X Error en paciente {patient_id}: {e}")

    except Exception as e:

        print(f"An unexpected error occurred: {e}")
 
print("--- Proceso finalizado ---")
 