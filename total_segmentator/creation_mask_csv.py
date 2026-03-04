import pandas as pd
import os
from pathlib import Path

# 1. Rutas de archivos
csv_original = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/updated_patients.csv'  # Cambia por el nombre real de tu CSV
csv_salida = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/updated_patients_mask.csv'
base_path_masks = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/segmentations/'

# 2. Cargar el CSV
df = pd.read_csv(csv_original)

# 3. Crear la columna 'mask' basada en el patient_id
# Usamos una función lambda para construir la ruta exacta para cada paciente
df['mask'] = df['patient_id'].apply(
    lambda x: os.path.join(base_path_masks, str(x), 'discos_secuenciales.nii.gz')
)

# 4. Reordenar las columnas para que 'mask' esté justo después de 'T1'
cols = list(df.columns)
# Buscamos el índice de T1
idx_t1 = cols.index('T1')

# Movemos la columna 'mask' (que está al final) a la posición idx_t1 + 1
cols.insert(idx_t1 + 1, cols.pop(cols.index('mask')))
df = df[cols]

# 5. Guardar el nuevo CSV
df.to_csv(csv_salida, index=False)

print(f"CSV generado con éxito: {csv_salida}")
print(df[['patient_id', 'T1', 'mask']].head()) # Visualización de control