PASOS PREVIOS PREPROCESAMIENTO:

1. python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/segmentation.py  : esto ejecuta el total segmentator y nos extrae las mascaras de los discos intervertebrales

2. python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/label_modification.py : esto reordena las mascaras de los discos (desde abajo hasta arriba) hasta el numero 5 (eliminando el resto de etiquetas >5)

1. python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/creation_mask_csv.py : se crea un csv a partir de otro que contiene la ruta de las imagenes y de los labels y se añade la ruta de las mask hechas.


2 OPCIONES: MACHINE LEARNING O DEEP LEARNING

- OPCION 1: MACHINE LEARNING : RADIOMICA

python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/radiomics_template/radiomics_extraction.py —input_csv xxxxx —mask_label X (ir modificando según cuantas etiquetas tenga tu imagen)

python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/train_binary.py : entrenamiento post radiomica

- OPCIÓN 2:  DEEP LEARNING: CROP

python /mnt/datalake/openmind/Midas/training_wheels_Spine/lib/my_lib/segmentation_nets/predict_nii_up/total_segmentator/cropped.py : esto recorta las imagenes por disco (5 imagenes en total) y crea un csv con la nueva ruta y por discos

python /mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/code/DL/densenet_training.py --csv /ruta/al/archivo.csv (creado en python anterior cropped.py)