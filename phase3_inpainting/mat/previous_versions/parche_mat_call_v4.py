#parche v4 para arreglar la llamada al Generator de MAT.
#
#error: Generator.forward() missing 1 required positional argument: 'c'
#causa: el Generator de MAT necesita 4 argumentos:
#  - img: imagen RGB enmascarada
#  - mask: mascara binaria
#  - z: vector latente (ruido para StyleGAN)
#  - c: vector de clase (label condicional, siempre cero para Places)
#actualmente el script solo pasa 3 argumentos y faltaba el label.
#
#solucion: replicar la llamada del script oficial generate_image.py:
#  z = torch.randn(1, G.z_dim, device=device)
#  c = torch.zeros(1, G.c_dim, device=device)
#  output = G(img, mask, z, c, truncation_psi=1, noise_mode='const')
#
#aplicar con: python parche_mat_call_v4.py

from pathlib import Path
import shutil

SCRIPT_PATH = Path("/home/pfc/cescuder/tfg/scripts/mat_inpainting.py")
BACKUP_PATH = SCRIPT_PATH.with_suffix(".py.bak4")

shutil.copy2(SCRIPT_PATH, BACKUP_PATH)
print(f"backup creado en {BACKUP_PATH}")

contenido = SCRIPT_PATH.read_text(encoding="utf-8")

viejo = '''            with torch.no_grad():
                #MAT recibe: imagen, mascara, label (0=Places)
                output = G(img_t, mask_t, torch.zeros([1], device=device).long())'''
nuevo = '''            with torch.no_grad():
                #MAT recibe 4 argumentos: imagen, mascara, vector latente z, vector de clase c.
                #para Places el label c es siempre cero (modelo no condicional por clases),
                #y z es ruido aleatorio que afecta los detalles de la sintesis StyleGAN-like.
                #truncation_psi=1 y noise_mode='const' son los valores del script oficial generate_image.py
                z = torch.randn(1, G.z_dim, device=device)
                c = torch.zeros(1, G.c_dim, device=device)
                output = G(img_t, mask_t, z, c, truncation_psi=1, noise_mode='const')'''
assert viejo in contenido, "no se encontro la llamada a G() tal como esperaba"
contenido = contenido.replace(viejo, nuevo, 1)

SCRIPT_PATH.write_text(contenido, encoding="utf-8")
print(f"parche aplicado a {SCRIPT_PATH}")
print("verificacion: deberian aparecer 'z = torch.randn' y 'truncation_psi=1'")
