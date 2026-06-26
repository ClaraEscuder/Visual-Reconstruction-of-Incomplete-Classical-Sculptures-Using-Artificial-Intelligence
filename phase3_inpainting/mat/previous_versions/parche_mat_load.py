#parche para arreglar la carga del .pkl de MAT.
#error original: _pickle.UnpicklingError: A load persistent id instruction was encountered,
#but no persistent_load function was specified.
#solucion: usar dnnlib.util.open_url en lugar de open() directamente, porque los .pkl de
#MAT/StyleGAN usan persistencia personalizada para serializar redes neuronales completas.
#
#aplicar con: python parche_mat_load.py
#esto edita mat_inpainting.py in-place haciendo backup .bak antes

from pathlib import Path
import shutil

SCRIPT_PATH = Path("/home/pfc/cescuder/tfg/scripts/mat_inpainting.py")
BACKUP_PATH = SCRIPT_PATH.with_suffix(".py.bak")

#hacer backup antes de tocar nada
shutil.copy2(SCRIPT_PATH, BACKUP_PATH)
print(f"backup creado en {BACKUP_PATH}")

contenido = SCRIPT_PATH.read_text(encoding="utf-8")

#cambio: sustituir el bloque de carga del .pkl por la version con dnnlib.util.open_url
viejo = '''    log.info(f"Loading MAT weights from: {PESOS_MAT}")
    with open(PESOS_MAT, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)'''
nuevo = '''    log.info(f"Loading MAT weights from: {PESOS_MAT}")
    #usar dnnlib.util.open_url para que la persistencia personalizada de los .pkl de MAT
    #funcione correctamente (open() directo no sabe desempaquetar redes neuronales serializadas)
    with dnnlib.util.open_url(str(PESOS_MAT)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)'''
assert viejo in contenido, "no se encontro el bloque de carga del .pkl en mat_inpainting.py"
contenido = contenido.replace(viejo, nuevo, 1)

SCRIPT_PATH.write_text(contenido, encoding="utf-8")
print(f"parche aplicado a {SCRIPT_PATH}")
print("verificacion: deberia aparecer 'dnnlib.util.open_url' al ejecutar 'grep open_url' sobre el script")
