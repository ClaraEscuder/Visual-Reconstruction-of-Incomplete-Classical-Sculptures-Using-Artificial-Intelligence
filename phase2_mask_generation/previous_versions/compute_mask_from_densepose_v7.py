#genera mascaras y cond.npz para los broken_body usando proyecciones anatomicas
#desde el cache de DensePose. la mascara puede provenir de tres estrategias:
#
#  1. simetria pixel-a-pixel: cuando la contraparte simetrica esta completa,
#     se reflejan los pixels exactos de su silueta. produce una mascara con la
#     forma real del miembro (no un bounding box).
#
#  2. proyeccion en cadena con silueta ahusada: para brazos y piernas sin
#     contraparte, se proyecta desde el torso (o desde la ultima region completa
#     de la cadena kinematica) con una silueta de seccion variable: ancho
#     proximal mayor que ancho distal. esto da forma de extremidad real en vez
#     de rectangulo.
#
#  3. proyeccion eliptica o ovalada: para la cabeza se usa una elipse simetrica
#     centrada; para mano/pie sin contraparte se usa una forma compacta ovalada.
#
#se escribe ademas un cond.npz por imagen con tres campos (I_pred, U_pred,
#V_pred) que combinan lo que se observa en el DensePose del cuerpo visible con
#lo que se proyecta para el miembro faltante. esto es el conditioning de 7
#canales que lee LaMa-v7 / v8.

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure


DIR_NPZ = Path("/home/pfc/cescuder/tfg/densepose_cache/broken_body")
DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/masks/broken_body_v7")


REGIONES = {
    "torso": [1, 2],
    "right_hand": [3],
    "left_hand": [4],
    "left_foot": [5],
    "right_foot":  [6],
    "upper_leg_R": [7, 9],
    "upper_leg_L": [8, 10],
    "lower_leg_R": [11, 13],
    "lower_leg_L": [12, 14],
    "upper_arm_L": [15, 17],
    "upper_arm_R": [16, 18],
    "lower_arm_L": [19, 21],
    "lower_arm_R": [20, 22],
    "head":  [23, 24],}


PART_ID_CANONICO = {region: caras[0] for region, caras in REGIONES.items()}
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

PORCENTAJE_VERTICAL = {
    "head": 0.13,
    "torso": 0.30,
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
    "left_foot":  0.07,
    "right_foot": 0.07,
}


UMBRAL_MUTILADA = 0.30
UMBRAL_COMPLETA = 0.80


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
    "torso":None,
    "head":None,
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

DIMENSIONES = {
    "left_foot": (0.10, 0.13),
    "right_foot": (0.10, 0.13),
    "left_hand":   (0.13, 0.10),
    "right_hand":  (0.13, 0.10),
    "lower_arm_L": (0.22, 0.10),
    "lower_arm_R": (0.22, 0.10),
    "upper_arm_L": (0.20, 0.11),
    "upper_arm_R": (0.20, 0.11),
    "lower_leg_L": (0.28, 0.12),
    "lower_leg_R": (0.28, 0.12),
    "upper_leg_L": (0.30, 0.14),
    "upper_leg_R": (0.30, 0.14),
    "head": (0.18, 0.14),
}

#factor de estrechamiento distal en la forma ahusada. los segmentos de brazo/
#pierna son casi cilindricos en la anatomia real, por eso solo aplica un
#estrechamiento muy suave (5%). la mano y el pie no usan esta forma sino
#"ovalada" (elipse compacta con bulto)
ANCHO_DISTAL_FACTOR = 0.95

DILATAR_PX_MIN = 10
DILATAR_PX_FACTOR = 60.0
MIN_PIX_PADRE = 30
SOLAPE_INTERNO_PCT = 0.04

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

    U = data["U"] if "U" in keys else np.zeros(body_parts.shape, dtype=np.float32)
    V = data["V"] if "V" in keys else np.zeros(body_parts.shape, dtype=np.float32)

    return body_parts.astype(np.uint8), np.asarray(U, dtype=np.float32), \
           np.asarray(V, dtype=np.float32), np.asarray(bbox).astype(int), score


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


def estado_region(region, coberturas):
    obs = coberturas.get(region, 0.0)
    esp = COBERTURA_ESPERADA.get(region, 0.0)
    if esp == 0:
        return "completa"
    ratio = obs / esp
    if ratio >= UMBRAL_COMPLETA:
        return "completa"
    if ratio >= UMBRAL_MUTILADA:
        return "munon"
    return "mutilada"


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


#estimacion del alto del cuerpo idealizado:
def estimar_alto_cuerpo(mascaras_por_region):
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

#proyeccion por simetria (espejado pixel-a-pixel):

def proyectar_por_simetria(region, mascaras_por_region, coberturas):
    sim = SIMETRICA.get(region)
    if sim is None:
        return None, "sin_simetrica"

    m_sim = mascaras_por_region.get(sim)
    if m_sim is None or m_sim.sum() < MIN_PIX_PADRE:
        return None, "simetrica_ausente"

    if estado_region(sim, coberturas) != "completa":
        return None, "simetrica_no_completa"

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

#utilidades de proyeccion (PCA del eje principal, punto distal):
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


def punto_extremo_distal(mascara, eje):
    ys, xs = np.where(mascara)
    pts = np.column_stack([ys, xs]).astype(np.float64)
    centroide = pts.mean(axis=0)
    proyecciones = (pts - centroide) @ eje
    umbral_top = np.percentile(proyecciones, 85)
    pts_distales = pts[proyecciones >= umbral_top]
    return pts_distales.mean(axis=0)



#silueta anatomica orientada (sustituye al rectangulo del v6):
def construir_silueta_orientada(punto_anclaje, eje, largo_px, ancho_px, shape,
                                forma="ahusada", ancho_distal_factor=ANCHO_DISTAL_FACTOR,
                                solape_interno=0.0):
    #forma "ahusada":  ancho varia linealmente de ancho_px (extremo proximal,
    #                  punto_anclaje) a ancho_px*ancho_distal_factor (extremo distal).
    #                  apropiado para brazos, piernas y cadenas multi-segmento.
    #forma "eliptica": elipse simetrica respecto al centro, mas ancha en el medio.
    #                  apropiada para cabeza.
    #forma "ovalada":  elipse compacta de baja excentricidad, apropiada para
    #                  mano y pie.

    largo_efectivo = largo_px + solape_interno
    centro_proy = punto_anclaje + eje * (largo_px / 2.0 - solape_interno / 2.0)
    perp = np.array([-eje[1], eje[0]])

    H, W = shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - centro_proy[0]
    dx = xx - centro_proy[1]
    proj_eje_grid = dy * eje[0] + dx * eje[1]
    proj_perp_grid = dy * perp[0] + dx * perp[1]

    #t normalizado de -0.5 (extremo proximal) a +0.5 (extremo distal)
    t = proj_eje_grid / largo_efectivo
    en_largo = np.abs(t) <= 0.5

    if forma == "ahusada":
        t_norm = (t + 0.5).clip(0.0, 1.0)
        ancho_local = ancho_px * (1.0 - (1.0 - ancho_distal_factor) * t_norm)
    elif forma == "eliptica":
        ancho_local = ancho_px * np.sqrt(np.clip(1.0 - (2.0 * t) ** 2, 0.0, 1.0))
    elif forma == "ovalada":
        ancho_local = ancho_px * np.sqrt(np.clip(1.0 - (2.0 * t) ** 2 * 0.6, 0.0, 1.0))
    else:
        ancho_local = ancho_px

    en_ancho = np.abs(proj_perp_grid) <= ancho_local / 2.0
    return en_largo & en_ancho



#proyeccion en cadena para brazos / piernas:
def es_brazo(region):
    return region in {"upper_arm_L", "upper_arm_R", "lower_arm_L", "lower_arm_R",
                      "left_hand", "right_hand"}


def es_pierna(region):
    return region in {"upper_leg_L", "upper_leg_R", "lower_leg_L", "lower_leg_R",
                      "left_foot", "right_foot"}


def es_mano_o_pie(region):
    return region in {"left_hand", "right_hand", "left_foot", "right_foot"}


def lado_de_la_region(region):
    if region.endswith("_L") or region == "left_hand" or region == "left_foot":
        return "L"
    if region.endswith("_R") or region == "right_hand" or region == "right_foot":
        return "R"
    return None


def cadena_brazo(lado):
    if lado == "L":
        return ["upper_arm_L", "lower_arm_L", "left_hand"]
    if lado == "R":
        return ["upper_arm_R", "lower_arm_R", "right_hand"]
    return []


def cadena_pierna(lado):
    if lado == "L":
        return ["upper_leg_L", "lower_leg_L", "left_foot"]
    if lado == "R":
        return ["upper_leg_R", "lower_leg_R", "right_foot"]
    return []


def localizar_hombro(mascara_torso, lado):
    ys_t, _ = np.where(mascara_torso)
    y_min = int(ys_t.min())
    y_max = int(ys_t.max())
    cuarto_y_max = y_min + (y_max - y_min) // 4
    H = mascara_torso.shape[0]
    mascara_arriba = mascara_torso & (np.arange(H)[:, None] <= cuarto_y_max)
    if mascara_arriba.sum() < MIN_PIX_PADRE:
        mascara_arriba = mascara_torso

    ys_s, xs_s = np.where(mascara_arriba)
    if lado == "L":
        idx = np.argmax(xs_s)
    else:
        idx = np.argmin(xs_s)
    return float(ys_s[idx]), float(xs_s[idx])


def localizar_cadera(mascara_torso, lado):
    ys_t, _ = np.where(mascara_torso)
    y_min = int(ys_t.min())
    y_max = int(ys_t.max())
    cuarto_y_min = y_max - (y_max - y_min) // 4
    H = mascara_torso.shape[0]
    mascara_abajo = mascara_torso & (np.arange(H)[:, None] >= cuarto_y_min)
    if mascara_abajo.sum() < MIN_PIX_PADRE:
        mascara_abajo = mascara_torso

    ys_s, xs_s = np.where(mascara_abajo)
    if lado == "L":
        idx = np.argmax(xs_s)
    else:
        idx = np.argmin(xs_s)
    return float(ys_s[idx]), float(xs_s[idx])


def proyectar_extremidad(region, mascaras_por_region, alto_referencia, coberturas):
    m_torso = mascaras_por_region.get("torso")
    if m_torso is None or m_torso.sum() < MIN_PIX_PADRE:
        return None, {"error": "torso_ausente"}

    lado = lado_de_la_region(region)
    if lado is None:
        return None, {"error": "lado_desconocido"}

    if es_brazo(region):
        cadena = cadena_brazo(lado)
        funcion_anclaje_proximal = localizar_hombro
        eje_proximal = np.array([0.3, 1.0]) if lado == "L" else np.array([0.3, -1.0])
    elif es_pierna(region):
        cadena = cadena_pierna(lado)
        funcion_anclaje_proximal = localizar_cadera
        eje_proximal = np.array([1.0, 0.15]) if lado == "L" else np.array([1.0, -0.15])
    else:
        return None, {"error": "region_no_extremidad"}

    eje_proximal = eje_proximal / np.linalg.norm(eje_proximal)

    if region not in cadena:
        return None, {"error": "region_no_en_cadena"}
    idx_region = cadena.index(region)

    idx_ultima_completa = -1
    for i in range(idx_region):
        if estado_region(cadena[i], coberturas) == "completa":
            idx_ultima_completa = i

    if idx_ultima_completa == -1:
        anc_y, anc_x = funcion_anclaje_proximal(m_torso, lado)
        punto_anclaje = np.array([anc_y, anc_x])
        eje = eje_proximal
        regiones_a_cubrir = cadena[: idx_region + 1]
        info_metodo = "desde_torso"
    else:
        m_ref = mascaras_por_region[cadena[idx_ultima_completa]]
        _, eje_ref, _ = calcular_eje_principal_pca(m_ref)
        if es_brazo(region):
            if lado == "L" and eje_ref[1] < 0:
                eje_ref = -eje_ref
            elif lado == "R" and eje_ref[1] > 0:
                eje_ref = -eje_ref
        else:
            if eje_ref[0] < 0:
                eje_ref = -eje_ref
        punto_anclaje = punto_extremo_distal(m_ref, eje_ref)
        eje = eje_ref
        regiones_a_cubrir = cadena[idx_ultima_completa + 1 : idx_region + 1]
        info_metodo = f"desde_{cadena[idx_ultima_completa]}"

    largo_total_pct = sum(DIMENSIONES[r][0] for r in regiones_a_cubrir)
    largo_px = float(largo_total_pct * alto_referencia)

    #usamos el ancho del segmento mas proximal como ancho de partida y el ancho
    #del segmento mas distal como ancho final. el ratio de los dos define el
    #grado de afilamiento real de la extremidad (calculado de DIMENSIONES, no
    #fijo): para upper+lower+hand sale ratio ~ 0.91 (casi cilindrico), para
    #upper_arm solo sale ratio = 1.0 (rectangulo redondeado)

    ancho_proximal_pct = DIMENSIONES[regiones_a_cubrir[0]][1]
    ancho_distal_pct   = DIMENSIONES[regiones_a_cubrir[-1]][1]
    ancho_px = float(ancho_proximal_pct * alto_referencia)
    factor_dinamico = max(0.6, min(1.0, ancho_distal_pct / max(ancho_proximal_pct, 1e-6)))
    solape_px = float(SOLAPE_INTERNO_PCT * alto_referencia)

    #si lo unico a cubrir es la mano o el pie, forma compacta ovalada (paleta).
    #si la cadena termina en mano/pie pero incluye tambien partes de
    #extremidad, usamos ahusada y luego el codigo de cond.npz pintara la zona
    #anyway. cualquier otra cadena usa ahusada con factor dinamico
    if len(regiones_a_cubrir) == 1 and es_mano_o_pie(regiones_a_cubrir[0]):
        forma = "ovalada"
    else:
        forma = "ahusada"

    mascara_proy = construir_silueta_orientada(
        punto_anclaje, eje, largo_px, ancho_px, m_torso.shape,
        forma=forma, ancho_distal_factor=factor_dinamico, solape_interno=solape_px,)

    info = {
        "metodo": info_metodo,
        "lado": lado,
        "regiones_cubiertas": regiones_a_cubrir,
        "punto_anclaje": punto_anclaje,
        "eje": eje,
        "largo_px": largo_px,
        "ancho_px": ancho_px,
        "ancho_distal_factor": factor_dinamico,
        "solape_px": solape_px,
        "forma": forma,}
    return mascara_proy, info


#proyeccion fallback para cabeza y otras regiones sin cadena distal:
def proyectar_fallback_anatomico(region, mascaras_por_region, alto_referencia):
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
            if eje[0] < 0:
                eje = -eje
    else:
        if eje[0] > 0:
            eje = -eje

    punto_anclaje = punto_extremo_distal(m_padre, eje)

    largo_pct, ancho_pct = DIMENSIONES[region]
    largo_px = float(largo_pct * alto_referencia)
    ancho_px = float(ancho_pct * alto_referencia)
    solape_px = float(SOLAPE_INTERNO_PCT * alto_referencia)

    #la cabeza usa elipse simetrica; mano o pie sin contraparte simetrica
    #usan ovalada compacta; cualquier otro fallback usa ahusada
    if region == "head":
        forma = "eliptica"
    elif es_mano_o_pie(region):
        forma = "ovalada"
    else:
        forma = "ahusada"

    mascara_proy = construir_silueta_orientada(
        punto_anclaje, eje, largo_px, ancho_px, m_padre.shape, forma=forma, solape_interno=solape_px,)

    info = {
        "metodo": "padre_abuelo",
        "padre": nombre_padre,
        "abuelo": nombre_abuelo,
        "centroide_padre": centroide_padre,
        "eje": eje,
        "punto_anclaje": punto_anclaje,
        "largo_px": largo_px,
        "ancho_px": ancho_px,
        "solape_px": solape_px,
        "forma": forma,
    }
    return mascara_proy, info


#sintesis de UV dentro de una region proyectada:
def sintetizar_uv_en_region(mascara_region: np.ndarray):
    #para una mascara binaria de una region proyectada, devuelve dos arrays U y V
    #(float32, mismo shape) que son gradientes lineales a lo largo del eje
    #principal de la region (U: 0 en el extremo proximal, 1 en el distal) y del
    #eje perpendicular (V: 0 a 1 cruzando el ancho). los pixels fuera de la
    #mascara quedan a 0. no es una parametrizacion SMPL real pero da al modelo
    #una nocion consistente de "donde dentro de la region estoy"
    U = np.zeros(mascara_region.shape, dtype=np.float32)
    V = np.zeros(mascara_region.shape, dtype=np.float32)

    ys, xs = np.where(mascara_region)
    if len(ys) < 2:
        return U, V

    pts = np.column_stack([ys, xs]).astype(np.float64)
    centroide = pts.mean(axis=0)
    pts_c = pts - centroide
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_p = np.argmax(eigvals)
    eje_principal = eigvecs[:, idx_p]
    eje_perp = np.array([-eje_principal[1], eje_principal[0]])

    proj_p = pts_c @ eje_principal
    proj_q = pts_c @ eje_perp

    p_min, p_max = proj_p.min(), proj_p.max()
    q_min, q_max = proj_q.min(), proj_q.max()
    if p_max - p_min > 1e-6:
        proj_p_norm = (proj_p - p_min) / (p_max - p_min)
    else:
        proj_p_norm = np.full_like(proj_p, 0.5)
    if q_max - q_min > 1e-6:
        proj_q_norm = (proj_q - q_min) / (q_max - q_min)
    else:
        proj_q_norm = np.full_like(proj_q, 0.5)

    U[ys, xs] = proj_p_norm.astype(np.float32)
    V[ys, xs] = proj_q_norm.astype(np.float32)
    return U, V


#pipeline completo------------------------------------------------------------------------
def generar_mascara_y_cond(npz_path):
    body_parts, U_orig, V_orig, bbox, score = cargar_predicciones(npz_path)
    H, W = body_parts.shape
    x0, y0, w, h = bbox

    mascaras = calcular_mascaras_por_region(body_parts)
    coberturas, pix_total = calcular_coberturas(mascaras)

    alto_referencia = estimar_alto_cuerpo(mascaras)
    if alto_referencia is None or alto_referencia < 50:
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
        return None, None, "sin_cuerpo_detectado", info

    mutiladas = identificar_mutiladas(coberturas)
    info["mutiladas"] = mutiladas

    if len(mutiladas) == 0:
        return None, None, "sin_mutiladas", info

    proyecciones_por_region = {}

    for region, obs, esp in mutiladas:
        m_proy, status_sim = proyectar_por_simetria(region, mascaras, coberturas)
        if m_proy is not None:
            proyecciones_por_region[region] = m_proy
            info["proyecciones"][region] = {"metodo": "simetria", "info": status_sim}
            continue

        if es_brazo(region) or es_pierna(region):
            m_proy, info_fb = proyectar_extremidad(region, mascaras, alto_referencia, coberturas)
        else:
            m_proy, info_fb = proyectar_fallback_anatomico(region, mascaras, alto_referencia)

        if m_proy is not None:
            proyecciones_por_region[region] = m_proy
            info["proyecciones"][region] = {"metodo": "fallback", "info": info_fb}
            continue

        info["proyecciones"][region] = {
            "metodo": "saltado",
            "info": {"sim": status_sim, "fb": info_fb},
        }

    if len(proyecciones_por_region) == 0:
        return None, None, "sin_proyecciones_validas", info

    cuerpo_presente = body_parts > 0
    union_proy = np.zeros((H, W), dtype=bool)
    for m in proyecciones_por_region.values():
        union_proy |= m
    union_pre = union_proy & ~cuerpo_presente

    if union_pre.sum() == 0:
        return None, None, "mascara_vacia_tras_restar_cuerpo", info

    dilatar_px = max(DILATAR_PX_MIN, int(round(alto_referencia / DILATAR_PX_FACTOR)))
    info["dilatar_px"] = dilatar_px

    estructura = generate_binary_structure(2, 2)
    mascara_final = binary_dilation(union_pre, structure=estructura, iterations=dilatar_px)
    mascara_final = mascara_final & ~cuerpo_presente

    I_pred = body_parts.copy()
    U_pred = U_orig.copy()
    V_pred = V_orig.copy()

    for region, m_proy in proyecciones_por_region.items():
        part_id = PART_ID_CANONICO.get(region, 0)
        zona_a_pintar = m_proy & ~cuerpo_presente & mascara_final
        if zona_a_pintar.sum() == 0:
            continue

        I_pred[zona_a_pintar] = part_id

        U_reg, V_reg = sintetizar_uv_en_region(zona_a_pintar)
        U_pred[zona_a_pintar] = U_reg[zona_a_pintar]
        V_pred[zona_a_pintar] = V_reg[zona_a_pintar]

    cond = {
        "mask": (mascara_final.astype(np.uint8) * 255),
        "I_pred": I_pred.astype(np.uint8),
        "U_pred": U_pred.astype(np.float16),
        "V_pred": V_pred.astype(np.float16),
    }

    return (mascara_final.astype(np.uint8) * 255), cond, "ok", info


#main----------------------------------------------------------------------
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    log_path = Path("/home/pfc/cescuder/tfg/logs/compute_mask_densepose_v7.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    log = logging.getLogger(__name__)

    npz_files = sorted(DIR_NPZ.glob("*.npz"))
    log.info(f"npz files encontrados: {len(npz_files)}")

    procesados, sin_proyeccion, errores = 0, 0, 0
    for npz_path in tqdm(npz_files, desc="masks v7"):
        try:
            stem = npz_path.stem
            #el cache de densepose guarda los archivos con el nombre completo
            #incluyendo la extension de imagen, por ejemplo "foo.jpg.npz".
            #en ese caso el stem es "foo.jpg" -> lo limpiamos para que coincida
            #con la convencion de los outputs de inferencia
            if stem.endswith(".jpg") or stem.endswith(".jpeg") or stem.endswith(".png") \
               or stem.endswith(".JPG") or stem.endswith(".JPEG") or stem.endswith(".PNG"):
                stem = Path(stem).stem

            mascara_uint8, cond, status, info = generar_mascara_y_cond(npz_path)

            if mascara_uint8 is None:
                sin_proyeccion += 1
                continue

            mask_path = DIR_SALIDA / f"{stem}_mask.png"
            Image.fromarray(mascara_uint8).save(mask_path)

            cond_path = DIR_SALIDA / f"{stem}_cond.npz"
            np.savez_compressed(cond_path, **cond)

            procesados += 1
        except Exception as e:
            errores += 1
            log.error(f"error en {npz_path.name}: {e}")

    log.info("COMPUTE MASK V7 COMPLETED")
    log.info(f"procesados:      {procesados}")
    log.info(f"sin proyeccion:  {sin_proyeccion}")
    log.info(f"errores:         {errores}")
    log.info(f"salida:          {DIR_SALIDA}")


if __name__ == "__main__":
    main()
