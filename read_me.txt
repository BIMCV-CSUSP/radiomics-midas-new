==========================================================================
PIPELINE DE PREPROCESAMIENTO Y ENTRENAMIENTO (SPINE SEGMENTATION)
==========================================================================

Este documento detalla los pasos necesarios para la segmentación de discos 
intervertebrales y el entrenamiento mediante Radiómica o Deep Learning.

--------------------------------------------------------------------------
1. PASOS PREVIOS: PREPROCESAMIENTO
--------------------------------------------------------------------------

PASO 1: Segmentación Automática
Ejecuta Total Segmentator para extraer las máscaras de los discos.
Comando:
python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/segmentation.py

PASO 2: Reordenación de Etiquetas
Reordena las máscaras de abajo hacia arriba (máximo 5 etiquetas, elimina >5).
Comando:
python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/label_modification.py

PASO 3: Creación de CSV de Máscaras
Genera un CSV integrando las rutas de imágenes, etiquetas originales y nuevas máscaras.
Comando:
python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/creation_mask_csv.py


--------------------------------------------------------------------------
2. OPCIONES DE ENTRENAMIENTO
--------------------------------------------------------------------------

OPCIÓN A: MACHINE LEARNING (RADIÓMICA)
--------------------------------------

A.1. Extracción de características:
(Modificar --input_csv y --mask_label según corresponda)
Comando:
python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/radiomics_template/radiomics_extraction.py --input_csv xxxxx --mask_label X

A.2. Entrenamiento:
Ejecuta el entrenamiento post-radiómica.
Comando:
python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/train_binary.py


OPCIÓN B: DEEP LEARNING (CROP)
------------------------------

B.1. Generación de recortes (Crops):
Recorta imágenes por disco (5 en total) y genera un nuevo CSV de rutas.
Comando:
python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/cropped.py

B.2. Entrenamiento DenseNet:
Usa el CSV generado en el paso anterior (B.1).
Comando:
python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/DL/densenet_training.py --csv /ruta/al/archivo_cropped.csv

==========================================================================