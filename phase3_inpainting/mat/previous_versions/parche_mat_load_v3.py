#parche v3 (corregido) para arreglar la carga de MAT.
#
#descubrimiento 1: el archivo Places_512_FullData_G.pkl es un state_dict plano (OrderedDict
#con tensores), NO un network pickle de StyleGAN. por eso legacy.load_network_pkl falla
#con UnpicklingError, esa funcion espera una red serializada completa.
#descubrimiento 2: el state_dict no contiene los parametros 'noise_strength' y 'noise_const'
#de las capas decoder, son parametros que el modelo inicializa por defecto. usamos
#strict=False en load_state_dict para ignorar esas missing keys (son 22, todas de noise).
#
#solucion: instanciar Generator manualmente con los parametros estandar de MAT-512 y luego
#cargar el state_dict con strict=False.
#
#aplicar con: python parche_mat_load_v3.py

from pathlib import Path
import shutil

SCRIPT_PATH = Path("/home/pfc/cescuder/tfg/scripts/mat_inpainting.py")
BACKUP_PATH = SCRIPT_PATH.with_suffix(".py.bak3")

shutil.copy2(SCRIPT_PATH, BACKUP_PATH)
print(f"backup creado en {BACKUP_PATH}")

contenido = SCRIPT_PATH.read_text(encoding="utf-8")

viejo = '''def cargar_mat(device):
    #añade el repo de MAT al path para poder importar sus modulos
    sys.path.insert(0, str(DIR_MAT_REPO))

    import dnnlib
    import legacy

    log.info(f"Loading MAT weights from: {PESOS_MAT}")
    #usar dnnlib.util.open_url para que la persistencia personalizada de los .pkl de MAT
    #funcione correctamente (open() directo no sabe desempaquetar redes neuronales serializadas)
    with dnnlib.util.open_url(str(PESOS_MAT)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    log.info("MAT loaded (frozen weights)")
    return G'''
nuevo = '''def cargar_mat(device):
    #añade el repo de MAT al path para poder importar sus modulos
    sys.path.insert(0, str(DIR_MAT_REPO))

    from networks.mat import Generator

    log.info(f"Loading MAT weights from: {PESOS_MAT}")
    #el .pkl es un state_dict plano (OrderedDict), no un network pickle completo.
    #instanciamos el Generator con los parametros estandar de MAT-512 y cargamos pesos
    #con strict=False porque el state_dict no incluye los parametros de noise (se inicializan por defecto)
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_MAT, img_channels=3)
    state_dict = torch.load(str(PESOS_MAT), map_location=device, weights_only=False)
    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    log.info(f"Generator created, state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    G = G.to(device).eval().requires_grad_(False)
    log.info("MAT loaded (frozen weights)")
    return G'''
assert viejo in contenido, "no se encontro la funcion cargar_mat tal como esperaba"
contenido = contenido.replace(viejo, nuevo, 1)

SCRIPT_PATH.write_text(contenido, encoding="utf-8")
print(f"parche aplicado a {SCRIPT_PATH}")
print("verificacion: deberia aparecer 'from networks.mat import Generator' y 'strict=False'")
