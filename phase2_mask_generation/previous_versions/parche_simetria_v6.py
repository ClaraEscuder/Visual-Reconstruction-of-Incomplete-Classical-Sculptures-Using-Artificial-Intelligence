#parche para v6: solo aceptar proyeccion por simetria si la region simetrica esta COMPLETA
#si es munon o mutilada, ir directamente a la proyeccion continua del v6 que sabe manejar esos casos correctamente
#
#aplicar con: python parche_simetria_v6.py
#esto edita compute_mask_from_densepose_v6.py in-place haciendo backup .bak antes

from pathlib import Path
import shutil

SCRIPT_PATH = Path("/home/pfc/cescuder/tfg/scripts/compute_mask_from_densepose_v6.py")
BACKUP_PATH = SCRIPT_PATH.with_suffix(".py.bak")

#hacer backup antes de tocar nada
shutil.copy2(SCRIPT_PATH, BACKUP_PATH)
print(f"backup creado en {BACKUP_PATH}")

contenido = SCRIPT_PATH.read_text(encoding="utf-8")

#cambio 1: anadir parametro coberturas a la firma de proyectar_por_simetria
viejo_1 = "def proyectar_por_simetria(region, mascaras_por_region):"
nuevo_1 = "def proyectar_por_simetria(region, mascaras_por_region, coberturas):"
assert viejo_1 in contenido, "no se encontro la firma de proyectar_por_simetria"
contenido = contenido.replace(viejo_1, nuevo_1, 1)

#cambio 2: anadir el check de estado al principio de la funcion proyectar_por_simetria
#justo despues de m_sim = mascaras_por_region.get(sim) bloque, anadir un nuevo check
viejo_2 = '''    m_sim = mascaras_por_region.get(sim)
    if m_sim is None or m_sim.sum() < MIN_PIX_PADRE:
        return None, "simetrica_ausente"'''
nuevo_2 = '''    m_sim = mascaras_por_region.get(sim)
    if m_sim is None or m_sim.sum() < MIN_PIX_PADRE:
        return None, "simetrica_ausente"

    #parche: solo usar simetria si la region simetrica esta COMPLETA.
    #si es munon (cobertura entre 30% y 80%) o mutilada, reflejarla daria un munon corto
    #en el otro lado, asi que es mejor ir al fallback continuo del v6 que sabe extender
    #la mascara desde el hombro/cadera con el largo correcto
    if estado_region(sim, coberturas) != "completa":
        return None, "simetrica_no_completa"'''
assert viejo_2 in contenido, "no se encontro el bloque m_sim de proyectar_por_simetria"
contenido = contenido.replace(viejo_2, nuevo_2, 1)

#cambio 3: actualizar la unica llamada a proyectar_por_simetria para pasar coberturas
viejo_3 = "m_proy, status_sim = proyectar_por_simetria(region, mascaras)"
nuevo_3 = "m_proy, status_sim = proyectar_por_simetria(region, mascaras, coberturas)"
assert viejo_3 in contenido, "no se encontro la llamada a proyectar_por_simetria"
contenido = contenido.replace(viejo_3, nuevo_3, 1)

SCRIPT_PATH.write_text(contenido, encoding="utf-8")
print(f"parche aplicado a {SCRIPT_PATH}")
print("verificacion: deberian aparecer los 3 cambios al ejecutar 'grep -n simetria_no_completa' sobre el script")
