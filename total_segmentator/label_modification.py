import nibabel as nib
import numpy as np
from scipy.ndimage import label, center_of_mass
from pathlib import Path

# Configuración de rutas
base_path = Path('/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/total_segmentator/segmentations/')

for patient_dir in base_path.iterdir():
    if patient_dir.is_dir():
        input_file = patient_dir / 'intervertebral_discs.nii.gz'
        output_file = patient_dir / 'discos_secuenciales.nii.gz'
        
        if not input_file.exists():
            continue

        print(f"Procesando: {patient_dir.name}")
        
        try:
            img = nib.load(str(input_file))
            data = img.get_fdata()
            affine = img.affine 

            # 1. Limpieza de ruido
            labeled, num = label(data > 0)
            sizes = np.bincount(labeled.ravel())
            mask_clean = np.zeros_like(labeled)
            for i in range(1, num + 1):
                if sizes[i] > 500: 
                    mask_clean[labeled == i] = 1

            # 2. Re-etiquetar
            labeled_clean, num_disks = label(mask_clean)
            if num_disks == 0: continue
                
            centers_voxel = center_of_mass(mask_clean, labeled_clean, range(1, num_disks + 1))

            # 3. CONVERSIÓN A COORDENADAS FÍSICAS (MM)
            # Esto ignora si el eje Z es + o - en el affine; nos da la posición real.
            info_discos = []
            for i, c_vox in enumerate(centers_voxel):
                c_real = nib.affines.apply_affine(affine, c_vox)
                # El índice [2] de c_real es SIEMPRE la altura física (Inferior -> Superior)
                info_discos.append({
                    'id_original': i + 1,
                    'z_fisica': c_real[2]
                })

            # 4. Ordenar de abajo hacia arriba (Z física menor a mayor)
            # El disco con la Z más baja (más cerca del sacro) será el 1.
            lista_ordenada = sorted(info_discos, key=lambda x: x['z_fisica'])

            # 5. Crear máscara final con los 5 discos inferiores
            data_final = np.zeros_like(data, dtype=np.int16)
            for nuevo_id, info in enumerate(lista_ordenada, start=1):
                if nuevo_id <= 5:
                    data_final[labeled_clean == info['id_original']] = nuevo_id
            
            new_img = nib.Nifti1Image(data_final, affine)
            nib.save(new_img, str(output_file))
            print(f"Éxito: {patient_dir.name} numerado correctamente (Basado en Z física).")

        except Exception as e:
            print(f"Error en {patient_dir.name}: {e}")

print("\n--- PROCESO COMPLETADO ---")