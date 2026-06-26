#genera mascaras de inpainting a partir de los campos densos extraidos por DensePose
#
#VERSION 5 - cambios respecto a v4.1:
#  - arreglo 1: alto_referencia ya no se toma del bbox de DensePose (que en esculturas
#    mutiladas es muy pequeno). en su lugar se estima el alto idealizado del cuerpo
#    completo a partir de las regiones presentes y sus porcentajes verticales conocidos.
#    se mide la extension vertical de cada region presente y se divide por su porcentaje
#    esperado en el cuerpo entero, y se toma la mediana de esas estimaciones como alto
#    de referencia. esto hace que las DIMENSIONES antropometricas se apliquen sobre
#    la altura real del cuerpo y no sobre un bbox mutilado.
#  - arreglo 1b: las DIMENSIONES de cada region se han ampliado un 30-50% para que las
#    proyecciones cubran tambien la zona donde el inpainting necesita margen, y la
#    DILATAR_PX ahora es proporcional al alto_referencia (mucho mas grande en imagenes
#    grandes que la dilatacion fija de 8 px del v4).
#  - arreglo 2: nueva logica especifica para brazos en proyectar_fallback_anatomico.
#    el padre del brazo es el torso, pero el centroide del torso esta en el eje
#    vertical, por lo que el PCA generico hacia "alto del torso" hacia que los brazos
#    se proyectaran verticalmente sobre la cabeza. ahora, cuando la region a proyectar
#    es un brazo (upper_arm_*, lower_arm_*, hand_*), el eje se fuerza en direccion
#    lateral horizontal (hacia la izquierda para brazo derecho de la imagen y viceversa),
#    el punto de anclaje se toma en el extremo lateral superior del torso del lado
#    correspondiente, y la proyeccion sale del hombro hacia fuera, que es lo
#    anatomicamente correcto.
#  - el resto del pipeline (filtro de descendiente, simetria, dilatacion, resta de
#    cuerpo) es identico al v4.1

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure


#configuracion de paths
DIR_NPZ = Path("/home/pfc/cescuder/tfg/densepose_cache/broken_body")
DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/masks/broken_body")


REGIONES = {
    "torso": [1, 2],
    "right_hand": [3],
    "left_hand": [4],
    "left_foot": [5],
    "right_foot": [6],
    "upper_leg_R": [7, 9],
    "upper_leg_L": [8, 10],
    "lower_leg_R": [11, 13],
    "lower_leg_L": [12, 14],
    "upper_arm_L": [15, 17],
    "upper_arm_R": [16, 18],
    "lower_arm_L": [19, 21],
    "lower_arm_R": [20, 22],
    "head": [23, 24],
}


COBERTURA_ESPERADA = {
    "torso": 0.180,
    "right_hand": 0.018,
    "left_hand": 0.018,
    "left_foot": 0.020,
    "right_foot": 0.020,
    "upper_leg_R": 0.060,
    "upper_leg_L": 0.060,
    "lower_leg_R": 0.050,
    "lower_leg_L": 0.050,
    "upper_arm_L": 0.040,
    "upper_arm_R": 0.040,
    "lower_arm_L": 0.040,
    "lower_arm_R": 0.040,
    "head": 0.060,
}


#porcentaje vertical aproximado de cada region respecto al alto del cuerpo entero,
#usado en estimar_alto_cuerpo para reconstruir el alto idealizado a partir de cualquier
#region presente. son aproximaciones antropometricas estandar
PORCENTAJE_VERTICAL = {
    "head": 0.13,   #cabeza ocupa ~13% del alto del cuerpo
    "torso": 0.30,   #torso ocupa ~30%
    "upper_leg_L": 0.24,
    "upper_leg_R": 0.24,
    "lower_leg_L": 0.22,
    "lower_leg_R": 0.22,
    "upper_arm_L": 0.16,
    "upper_arm_R": 0.16,
    "lower_arm_L": 0.16,
    "lower_arm_R": 0.16,
    "left_hand": 0.10,
    "right_hand": 0.10,
    "left_foot": 0.07,
    "right_foot": 0.07,
}


UMBRAL_MUTILADA = 0.30


SIMETRICA = {
    "right_hand": "left_hand",
    "left_hand": "right_hand",
    "left_foot": "right_foot",
    "right_foot": "left_foot",
    "upper_leg_R": "upper_leg_L",
    "upper_leg_L": "upper_leg_R",
    "lower_leg_R": "lower_leg_L",
    "lower_leg_L": "lower_leg_R",
    "upper_arm_L": "upper_arm_R",
    "upper_arm_R": "upper_arm_L",
    "lower_arm_L": "lower_arm_R",
    "lower_arm_R": "lower_arm_L",
    "torso": None,
    "head": None,
}


PADRE = {
    "left_foot": "lower_leg_L",
    "right_foot": "lower_leg_R",
    "lower_leg_L": "upper_leg_L",
    "lower_leg_R": "upper_leg_R",
    "upper_leg_L": "torso",
    "upper_leg_R": "torso",
    "left_hand": "lower_arm_L",
    "right_hand": "lower_arm_R",
    "lower_arm_L": "upper_arm_L",
    "lower_arm_R": "upper_arm_R",
    "upper_arm_L": "torso",
    "upper_arm_R": "torso",
    "head": "torso",
}


HIJOS = {}
for _hijo, _padre in PADRE.items():
    HIJOS.setdefault(_padre, []).append(_hijo)


ABUELO = {
    "left_foot": "upper_leg_L",
    "right_foot": "upper_leg_R",
    "lower_leg_L": "torso",
    "lower_leg_R": "torso",
    "left_hand": "upper_arm_L",
    "right_hand": "upper_arm_R",
    "lower_arm_L": "torso",
    "lower_arm_R": "torso",
    "head": None,
}


#dimensiones antropometricas de cada region a proyectar, expresadas como
#(largo_pct, ancho_pct) sobre el alto del cuerpo entero (no del bbox).
#valores ampliados respecto al v4 para dar mas margen al inpainting:
#porcentaje verticales antropometricos + ~30-50% extra para que el inpainting tenga aire
DIMENSIONES = {
    "left_foot": (0.10, 0.13),   #pie tumbado, mas ancho que alto
    "right_foot": (0.10, 0.13),
    "left_hand": (0.13, 0.10),
    "right_hand": (0.13, 0.10),
    "lower_arm_L": (0.22, 0.10),   #antebrazo: 22% alto cuerpo (era 16%, ahora con margen)
    "lower_arm_R": (0.22, 0.10),
    "upper_arm_L": (0.20, 0.11),   #brazo superior: 20% alto cuerpo
    "upper_arm_R": (0.20, 0.11),
    "lower_leg_L": (0.28, 0.12),   #pierna inferior: 28% alto cuerpo
    "lower_leg_R": (0.28, 0.12),
    "upper_leg_L": (0.30, 0.14),   #muslo: 30% alto cuerpo
    "upper_leg_R": (0.30, 0.14),
    "head": (0.18, 0.14),
}


#dilatacion proporcional al alto_referencia. en una imagen grande (alto 1500px)
#dilata unos 25 pixeles, en una pequena (alto 600px) unos 10 pixeles.
#se aplica como max(DILATAR_PX_MIN, alto_ref / DILATAR_PX_FACTOR)
DILATAR_PX_MIN = 10
DILATAR_PX_FACTOR = 60.0


MIN_PIX_PADRE = 30



#carga y preparacion:

def cargar_predicciones(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    keys = list(data.keys())

    body_parts = None
    for k in ["body_parts", "parts", "labels", "I", "S"]:
        if k in keys:
            body_parts = data[k]
            break
    if body_parts is None:
        raise ValueError(f"no se encontro array de body_parts en {npz_path}, claves: {keys}")

    bbox = None
    for k in ["bbox", "box"]:
        if k in keys:
            bbox = data[k]
            break
    if bbox is None:
        raise ValueError(f"no se encontro bbox en {npz_path}")

    score = float(data["score"].item()) if "score" in keys else None

    return body_parts.astype(np.uint8), np.asarray(bbox).astype(int), score


def calcular_mascaras_por_region(body_parts):
    mascaras = {}
    for region, caras in REGIONES.items():
        m = np.zeros(body_parts.shape, dtype=bool)
        for cara in caras:
            m |= (body_parts == cara)
        mascaras[region] = m
    return mascaras


def calcular_coberturas(mascaras_por_region):
    pix_total = sum(int(m.sum()) for m in mascaras_por_region.values())
    if pix_total == 0:
        return {r: 0.0 for r in mascaras_por_region}, 0
    coberturas = {r: int(m.sum()) / pix_total for r, m in mascaras_por_region.items()}
    return coberturas, pix_total


def tiene_descendiente_presente(region, coberturas):
    hijos_directos = HIJOS.get(region, [])
    for hijo in hijos_directos:
        esp = COBERTURA_ESPERADA.get(hijo, 0.0)
        if esp == 0.0:
            continue
        obs = coberturas.get(hijo, 0.0)
        if obs >= UMBRAL_MUTILADA * esp:
            return True
        if tiene_descendiente_presente(hijo, coberturas):
            return True
    return False


def identificar_mutiladas(coberturas):
    mutiladas = []
    for region, obs in coberturas.items():
        esp = COBERTURA_ESPERADA.get(region, 0.0)
        if esp == 0.0:
            continue
        if obs < UMBRAL_MUTILADA * esp:
            if tiene_descendiente_presente(region, coberturas):
                continue
            mutiladas.append((region, obs, esp))
    return mutiladas



#estimacion del alto del cuerpo idealizado (arreglo)_

def estimar_alto_cuerpo(mascaras_por_region):
    #estima el alto del cuerpo entero idealizado a partir de las regiones presentes.
    #para cada region presente se mide su extension vertical (max_y - min_y), se divide
    #por su porcentaje vertical antropometrico esperado, y se obtiene una estimacion
    #del alto total. devuelve la mediana de esas estimaciones como valor robusto.
    estimaciones = []
    for region, mask in mascaras_por_region.items():
        if mask.sum() < MIN_PIX_PADRE:
            continue
        pct = PORCENTAJE_VERTICAL.get(region)
        if pct is None or pct <= 0:
            continue
        ys, _ = np.where(mask)
        extension_vertical = float(ys.max() - ys.min() + 1)
        if extension_vertical < 5:
            continue
        estimaciones.append(extension_vertical / pct)

    if len(estimaciones) == 0:
        return None
    return float(np.median(estimaciones))



#proyeccion por simetria:

def proyectar_por_simetria(region, mascaras_por_region):
    sim = SIMETRICA.get(region)
    if sim is None:
        return None, "sin_simetrica"

    m_sim = mascaras_por_region.get(sim)
    if m_sim is None or m_sim.sum() < MIN_PIX_PADRE:
        return None, "simetrica_ausente"

    union_cuerpo = np.zeros_like(m_sim, dtype=bool)
    for m in mascaras_por_region.values():
        union_cuerpo |= m
    if union_cuerpo.sum() == 0:
        return None, "sin_cuerpo"

    ys, xs = np.where(union_cuerpo)
    eje_x = int(round(xs.mean()))

    H, W = m_sim.shape
    m_espejada = np.zeros_like(m_sim, dtype=bool)
    ys_s, xs_s = np.where(m_sim)
    xs_nuevos = 2 * eje_x - xs_s
    valido = (xs_nuevos >= 0) & (xs_nuevos < W)
    m_espejada[ys_s[valido], xs_nuevos[valido]] = True

    return m_espejada, "ok"



#proyeccion fallback anatomico (con arreglo 2 para BRAZOS!!!!!!!!!!!111)

def calcular_eje_principal_pca(mascara):
    ys, xs = np.where(mascara)
    pts = np.column_stack([ys, xs]).astype(np.float64)
    centroide = pts.mean(axis=0)
    if len(pts) < 2:
        return centroide, np.array([1.0, 0.0]), np.array([0.0, 0.0])
    pts_c = pts - centroide
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_principal = np.argmax(eigvals)
    eje = eigvecs[:, idx_principal]
    return centroide, eje, eigvals


def es_brazo(region):
    #devuelve True si la region pertenece a la cadena anatomica del brazo
    return region in {"upper_arm_L", "upper_arm_R", "lower_arm_L", "lower_arm_R",
                      "left_hand", "right_hand"}


def lado_del_brazo(region):
    #L significa lado izquierdo de la persona = lado derecho de la imagen (mirando de frente)
    #R significa lado derecho de la persona = lado izquierdo de la imagen
    if region.endswith("_L"):
        return "L"
    if region.endswith("_R"):
        return "R"
    if region == "left_hand":
        return "L"
    if region == "right_hand":
        return "R"
    return None


def proyectar_brazo_lateral(region, mascaras_por_region, alto_referencia):
    #proyeccion especifica para brazos (arreglo 2 del v5).
    #en vez de usar el PCA del torso (que sale vertical) y la lineas padre-abuelo
    #(que tambien queda vertical), aprovechamos que un brazo crece lateralmente desde
    #el extremo del torso del lado correspondiente.
    #algoritmo:
    #  1. localizar el extremo lateral superior del torso (hombro) del lado del brazo
    #  2. fijar la direccion del eje en horizontal hacia fuera del cuerpo
    #  3. dibujar el rectangulo orientado de las dimensiones esperadas

    m_torso = mascaras_por_region.get("torso")
    if m_torso is None or m_torso.sum() < MIN_PIX_PADRE:
        return None, {"error": "torso_ausente"}

    lado = lado_del_brazo(region)
    if lado is None:
        return None, {"error": "lado_desconocido"}

    ys_t, xs_t = np.where(m_torso)
    #localizar el hombro como punto extremo lateral del cuarto superior del torso.
    #cuarto superior porque el hombro esta arriba, no en la mitad ni abajo
    y_min = int(ys_t.min())
    y_max = int(ys_t.max())
    cuarto_y_max = y_min + (y_max - y_min) // 4
    mascara_arriba = m_torso & (np.arange(m_torso.shape[0])[:, None] <= cuarto_y_max)
    if mascara_arriba.sum() < MIN_PIX_PADRE:
        mascara_arriba = m_torso  #fallback si el cuarto superior es muy pequeno

    ys_s, xs_s = np.where(mascara_arriba)
    if lado == "L":
        #brazo izquierdo de la persona = lado derecho de la imagen, x mayor
        idx = np.argmax(xs_s)
    else:
        #brazo derecho de la persona = lado izquierdo de la imagen, x menor
        idx = np.argmin(xs_s)
    hombro_y = float(ys_s[idx])
    hombro_x = float(xs_s[idx])

    #direccion lateral horizontal: hacia fuera del cuerpo
    if lado == "L":
        eje = np.array([0.0, 1.0])   #x crece (hacia derecha de imagen)
    else:
        eje = np.array([0.0, -1.0])  #x decrece

    #anadir un componente vertical pequeno hacia abajo para que el brazo caiga ligeramente
    #(brazos relajados cuelgan, no salen perfectamente horizontales).
    #con peso 0.3 hacia abajo, da un angulo de unos 17 grados respecto a la horizontal,
    #que es razonable para esculturas estaticas
    eje[0] += 0.3
    eje = eje / np.linalg.norm(eje)

    largo_pct, ancho_pct = DIMENSIONES[region]
    largo_px = float(largo_pct * alto_referencia)
    ancho_px = float(ancho_pct * alto_referencia)

    #para los antebrazos y manos, el ancla no es el hombro sino el extremo del segmento padre.
    #si el brazo superior tambien esta presente, usamos su extremo distal como ancla.
    #si no, partimos del hombro
    nombre_padre = PADRE.get(region)
    m_padre = mascaras_por_region.get(nombre_padre) if nombre_padre else None
    if m_padre is not None and m_padre.sum() >= MIN_PIX_PADRE and nombre_padre != "torso":
        #usar el extremo distal del padre como ancla
        ys_p, xs_p = np.where(m_padre)
        pts_p = np.column_stack([ys_p, xs_p]).astype(np.float64)
        centroide_p = pts_p.mean(axis=0)
        proyecciones = (pts_p - centroide_p) @ eje
        umbral_top = np.percentile(proyecciones, 85)
        pts_distales = pts_p[proyecciones >= umbral_top]
        punto_anclaje = pts_distales.mean(axis=0)
    else:
        punto_anclaje = np.array([hombro_y, hombro_x])

    centro_proy = punto_anclaje + eje * (largo_px / 2.0)
    perp = np.array([-eje[1], eje[0]])

    H, W = m_torso.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - centro_proy[0]
    dx = xx - centro_proy[1]
    proj_eje_grid  = dy * eje[0]  + dx * eje[1]
    proj_perp_grid = dy * perp[0] + dx * perp[1]
    mascara_proy = (np.abs(proj_eje_grid) <= largo_px / 2.0) & \
                   (np.abs(proj_perp_grid) <= ancho_px / 2.0)

    info = {
        "metodo": "lateral_brazo",
        "lado": lado,
        "punto_anclaje": punto_anclaje,
        "eje": eje,
        "largo_px": largo_px,
        "ancho_px": ancho_px,
    }
    return mascara_proy, info


def proyectar_fallback_anatomico(region, mascaras_por_region, alto_referencia):
    #para brazos: usa la logica especifica lateral del arreglo 2
    if es_brazo(region):
        return proyectar_brazo_lateral(region, mascaras_por_region, alto_referencia)

    #para piernas, pies y cabeza: la logica padre-abuelo del v4 funciona bien
    #(las piernas si crecen verticalmente desde la cadera, y la cabeza desde el cuello)
    nombre_padre = PADRE.get(region)
    if nombre_padre is None:
        return None, "sin_padre_definido"

    m_padre = mascaras_por_region.get(nombre_padre)
    if m_padre is None or m_padre.sum() < MIN_PIX_PADRE:
        return None, f"padre_{nombre_padre}_ausente"

    centroide_padre, eje, _ = calcular_eje_principal_pca(m_padre)

    nombre_abuelo = ABUELO.get(region)
    if nombre_abuelo is not None:
        m_abuelo = mascaras_por_region.get(nombre_abuelo)
        if m_abuelo is not None and m_abuelo.sum() >= MIN_PIX_PADRE:
            ys_a, xs_a = np.where(m_abuelo)
            centroide_abuelo = np.array([ys_a.mean(), xs_a.mean()])
            dir_distal = centroide_padre - centroide_abuelo
            if np.dot(dir_distal, eje) < 0:
                eje = -eje
        else:
            #sin abuelo: para piernas y pies, distal es hacia abajo (y crece)
            if eje[0] < 0:
                eje = -eje
    else:
        #cabeza: padre es torso, distal es hacia arriba (y decrece)
        if eje[0] > 0:
            eje = -eje

    ys, xs = np.where(m_padre)
    pts = np.column_stack([ys, xs]).astype(np.float64)
    proyecciones = (pts - centroide_padre) @ eje
    umbral_top = np.percentile(proyecciones, 85)
    pts_distales = pts[proyecciones >= umbral_top]
    punto_anclaje = pts_distales.mean(axis=0)

    largo_pct, ancho_pct = DIMENSIONES[region]
    largo_px = float(largo_pct * alto_referencia)
    ancho_px = float(ancho_pct * alto_referencia)

    centro_proy = punto_anclaje + eje * (largo_px / 2.0)
    perp = np.array([-eje[1], eje[0]])

    H, W = m_padre.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - centro_proy[0]
    dx = xx - centro_proy[1]
    proj_eje_grid  = dy * eje[0]  + dx * eje[1]
    proj_perp_grid = dy * perp[0] + dx * perp[1]
    mascara_proy = (np.abs(proj_eje_grid) <= largo_px / 2.0) & \
                   (np.abs(proj_perp_grid) <= ancho_px / 2.0)

    info = {
        "metodo": "padre_abuelo",
        "padre": nombre_padre,
        "abuelo": nombre_abuelo,
        "centroide_padre": centroide_padre,
        "eje": eje,
        "punto_anclaje": punto_anclaje,
        "centro_proy": centro_proy,
        "largo_px": largo_px,
        "ancho_px": ancho_px,
    }
    return mascara_proy, info



#pipeline completo:

def generar_mascara_inpainting(npz_path):
    body_parts, bbox, score = cargar_predicciones(npz_path)
    H, W = body_parts.shape
    x0, y0, w, h = bbox

    mascaras = calcular_mascaras_por_region(body_parts)
    coberturas, pix_total = calcular_coberturas(mascaras)

    #ARREGLO 1: alto idealizado del cuerpo entero, no del bbox
    alto_referencia = estimar_alto_cuerpo(mascaras)
    if alto_referencia is None or alto_referencia < 50:
        #fallback al bbox si la estimacion fallo (caso extremo, casi nunca pasa)
        alto_referencia = float(max(int(h), int(w)))

    info = {
        "score": score,
        "bbox": [int(x0), int(y0), int(w), int(h)],
        "alto_referencia": float(alto_referencia),
        "pix_total": pix_total,
        "coberturas": coberturas,
        "proyecciones": {},
    }

    if pix_total == 0:
        return None, "sin_cuerpo_detectado", info

    mutiladas = identificar_mutiladas(coberturas)
    info["mutiladas"] = mutiladas

    if len(mutiladas) == 0:
        return None, "sin_mutiladas", info

    union_proy = np.zeros((H, W), dtype=bool)
    for region, obs, esp in mutiladas:
        m_proy, status_sim = proyectar_por_simetria(region, mascaras)
        if m_proy is not None:
            union_proy |= m_proy
            info["proyecciones"][region] = {"metodo": "simetria", "info": status_sim}
            continue

        m_proy, info_fb = proyectar_fallback_anatomico(region, mascaras, alto_referencia)
        if m_proy is not None:
            union_proy |= m_proy
            info["proyecciones"][region] = {"metodo": "fallback", "info": info_fb}
            continue

        info["proyecciones"][region] = {
            "metodo": "saltado",
            "info": {"sim": status_sim, "fb": info_fb},
        }

    if union_proy.sum() == 0:
        return None, "sin_proyecciones_validas", info

    cuerpo_presente = body_parts > 0
    union_pre = union_proy & ~cuerpo_presente

    if union_pre.sum() == 0:
        return None, "mascara_vacia_tras_restar_cuerpo", info

    #ARREGLO 1b: dilatacion proporcional al alto_referencia
    dilatar_px = max(DILATAR_PX_MIN, int(round(alto_referencia / DILATAR_PX_FACTOR)))
    info["dilatar_px"] = dilatar_px

    estructura = generate_binary_structure(2, 2)
    mascara_final = binary_dilation(union_pre, structure=estructura, iterations=dilatar_px)
    mascara_final = mascara_final & ~cuerpo_presente

    return (mascara_final.astype(np.uint8) * 255), "ok", info


#maiN==========================================================

def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    log_path = Path("/home/pfc/cescuder/tfg/logs/compute_mask_densepose_v5.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],)
    log = logging.getLogger(__name__)

    npz_files = sorted(DIR_NPZ.glob("*.npz"))
    log.info(f"encontrados {len(npz_files)} archivos npz en {DIR_NPZ}")

    procesadas = 0
    errores = 0
    contadores_status = {}

    for npz_path in tqdm(npz_files, desc="generating masks v5"):
        stem = npz_path.stem
        mask_path = DIR_SALIDA / f"{stem}_mask.png"
        if mask_path.exists():
            procesadas += 1
            continue

        try:
            mascara, status, info = generar_mascara_inpainting(npz_path)
            contadores_status[status] = contadores_status.get(status, 0) + 1
            if mascara is not None:
                Image.fromarray(mascara).save(mask_path)
                procesadas += 1
        except Exception as e:
            log.warning(f"error en {npz_path.name}: {e}")
            errores += 1

    log.info("mask generation v5 completed")
    log.info(f"  procesadas: {procesadas}")
    for status, count in contadores_status.items():
        log.info(f"  status ({status}): {count}")
    log.info(f"  errores: {errores}")
    log.info(f"  output: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
