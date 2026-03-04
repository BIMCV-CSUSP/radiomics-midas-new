#!/usr/bin/env python3
import subprocess
from pathlib import Path

BASE = Path("/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/xnat_downloads")
CONVERTER = "convert2nifti.py"  # ajusta si está en otra ruta

#P_XXXXX/*/assessors/*/resources/SEG/*.dcm
dcm_files = list(BASE.glob("*/*/*"))
# seg_files = list(BASE.glob("P_*/**/assessors/*/resources/SEG/*.dcm"))

if not dcm_files:
    print("No se encontraron SEG .dcm. Revisa el patrón o las rutas.")
    raise SystemExit(1)

for dcm in dcm_files:
    patient = next(p for p in dcm.parts if p.startswith("XNAT_"))
    subject = dcm.parts[-3]
    seq = dcm.parts[-2]
    image = dcm.parts[-1]
    out_dir = BASE / "outputs" / patient / subject / seq

    # usa el nombre del assessor para separar salidas
    # busca el archivo de segmentación correspondiente
    # seg_file = None
    # for seg in seg_files:
    #     if patient in seg.parts:
    #         seg_file = seg
    #         break

    # print("Seg:", seg_file)
    # try:
    #     assessor_idx = dcm.parts.index("assessors") + 1
    #     assessor = dcm.parts[assessor_idx]
    #     print(assessor)
    # except ValueError:
    #     assessor = "unknown_assessor"

    # out_dir = BASE / "outputs" / patient / seg_file.parent.name if seg_file else "no_seg"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # # print("Output dir:", out_dir)
    cmd = [
        "python", CONVERTER,
        "--image_dir", str(dcm),
        # "--seg_file", str(seg_file),
        "--output_dir", str(out_dir),
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
