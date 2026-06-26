#genera mascaras de inpainting a partir de los campos densos extraidos por DensePose.
#el pipeline lee los .npz que produjo extract_densepose.py, agrupa las 24 caras de la malla SMPL en 14 regiones anatomicas, y compara la cobertura observada por region (pixeles_region / pixeles_totales_cuerpo) con una cobertura esperada calibrada sobre imagenes COCO.
#una region se marca como mutilada cuando su cobertura observada cae por debajo del 30% de la esperada y ningun descendiente en su cadena anatomica esta presente. el filtro de descendiente es importante porque DensePose a veces clasifica mal una region pero detecta correctamente a sus hijos (por ejemplo, detecta la mano pero pierde el antebrazo en una pose poco habitual). en ese caso la region no esta mutilada de verdad sino mal segmentada, y por anatomia (no se puede tener una mano sin antebrazo) la descartamos.
#para cada region mutilada se intenta primero proyectar por simetria espejando la mascara de la region simetrica respecto al centro horizontal del cuerpo. si la simetrica tampoco esta disponible se cae al fallback anatomico: PCA de la mascara del padre para obtener su eje principal, orientacion del eje en sentido distal usando el abuelo (o "hacia arriba" cuando la region es la cabeza), busqueda del extremo distal del padre como mediana del 15% de pixeles con mayor proyeccion sobre el eje, y dibujado de un rectangulo orientado de dimensiones antropometricas estandar desde ese extremo en la direccion del eje.
#la union de todas las proyecciones se resta del cuerpo presente para no pintar sobre lo que ya esta y se dilata DILATAR_PX pixeles para dar margen al inpainting. el resultado se guarda como PNG binario.

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure


DIR_NPZ = Path("/home/pfc/cescuder/tfg/densepose_cache/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/masks/broken_body")


#mapeo de las 24 caras de la malla SMPL a 14 regiones anatomicas (ver paper de DensePose, Guler et al. 2018, figura 2)
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


#fraccion del cuerpo que cada region ocupa en una persona completa tipica, calibrada a partir de la cobertura media en imagenes COCO con DensePose
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


#una region se considera mutilada si su cobertura observada cae por debajo de UMBRAL_MUTILADA por la esperada
UMBRAL_MUTILADA = 0.30


#par simetrico de cada region (None si no tiene)
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


#padre anatomico en la cadena cinematica del cuerpo. se usa en el fallback geometrico: si una region esta rota y su simetrica tambien, proyectamos la region desde su padre (los pies desde la pantorrilla, la cabeza desde el torso, etc.)
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


#inverso de PADRE: para cada region, lista de regiones cuyo padre es esta. usado en tiene_descendiente_presente para chequear recursivamente si algun hijo, nieto, etc. esta presente
HIJOS = {}
for _hijo, _padre in PADRE.items():
    HIJOS.setdefault(_padre, []).append(_hijo)


#abuelo anatomico, usado para orientar el sentido distal del eje principal del padre durante el fallback geometrico. la cabeza no tiene abuelo y usa "hacia arriba" como direccion canonica
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


#dimensiones antropometricas estandar de cada region a proyectar, expresadas como (largo_pct, ancho_pct) sobre la dimension mayor del bounding box. valores aproximados: pie ~14% altura, mano ~10%, cabeza ~13% altura
DIMENSIONES = {
    "left_foot": (0.14, 0.10),
    "right_foot": (0.14, 0.10),
    "left_hand": (0.10, 0.07),
    "right_hand": (0.10, 0.07),
    "lower_arm_L": (0.16, 0.06),
    "lower_arm_R": (0.16, 0.06),
    "upper_arm_L": (0.16, 0.07),
    "upper_arm_R": (0.16, 0.07),
    "lower_leg_L": (0.22, 0.08),
    "lower_leg_R": (0.22, 0.08),
    "upper_leg_L": (0.24, 0.10),
    "upper_leg_R": (0.24, 0.10),
    "head": (0.13, 0.10),}


#dilatacion final aplicada a la mascara para dar margen al inpainting
DILATAR_PX = 8

#minimo de pixeles para considerar una mascara util en el calculo del eje
MIN_PIX_PADRE = 30


def cargar_predicciones(npz_path):
    #carga el .npz generado por extract_densepose.py. se espera un array body_parts (HxW, valores 0-24), bbox [x,y,w,h] y score, con tolerancia a otros nombres comunes de campo
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

    score = float(data["score"]) if "score" in keys else None

    return body_parts.astype(np.uint8), np.asarray(bbox).astype(int), score


def calcular_mascaras_por_region(body_parts):
    #convierte el array de caras DensePose en un dict region -> mascara binaria HxW
    mascaras = {}
    for region, caras in REGIONES.items():
        m = np.zeros(body_parts.shape, dtype=bool)
        for cara in caras:
            m |= (body_parts == cara)
        mascaras[region] = m
    return mascaras


def calcular_coberturas(mascaras_por_region):
    #para cada region calcula su cobertura como pixeles_region / pixeles_totales_cuerpo. el denominador es invariante al encuadre y al tamano de la imagen
    pix_total = sum(int(m.sum()) for m in mascaras_por_region.values())
    if pix_total == 0:
        return {r: 0.0 for r in mascaras_por_region}, 0
    coberturas = {r: int(m.sum()) / pix_total for r, m in mascaras_por_region.items()}
    return coberturas, pix_total


def tiene_descendiente_presente(region, coberturas):
    #devuelve True si algun descendiente (hijo, nieto, ...) de region esta presente.
    #se usa para filtrar misclasificaciones de DensePose: si la mano esta detectada pero el antebrazo "falta", el antebrazo no falta de verdad, simplemente DensePose lo segmento como otra cosa. anatomicamente no se puede tener una mano sin antebrazo, asi que descartamos la mutilacion.
    #el chequeo es recursivo: descenderemos por toda la cadena de hijos hasta encontrar uno presente o agotar el arbol
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
    #una region es mutilada si su cobertura observada cae por debajo del umbral relativo a la esperada y ademas ningun descendiente en su cadena anatomica esta presente. el segundo filtro descarta misclasificaciones de DensePose
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


def proyectar_por_simetria(region, mascaras_por_region):
    #proyecta espejando la mascara de la region simetrica respecto al centro horizontal del cuerpo. funciona bien cuando la pose es razonablemente simetrica (estatuas estaticas). cuando la pose es asimetrica el resultado puede caer fuera de su sitio, pero el filtro de descendiente captura los casos mas problematicos antes de llegar aqui
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


def calcular_eje_principal_pca(mascara):
    #primer componente PCA de una mascara binaria. devuelve (centroide, eje_unitario, eigenvalues) en formato (y, x) donde y crece hacia abajo
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


def proyectar_fallback_anatomico(region, mascaras_por_region, alto_referencia):
    #proyecta una region desde su padre anatomico cuando ni la region ni su simetrica estan disponibles.
    #algoritmo: PCA del padre da el eje principal del miembro, el abuelo orienta el sentido distal (vector centroide_abuelo -> centroide_padre indica hacia donde "crece" el miembro), el extremo distal del padre se localiza como mediana del 15% de pixeles con mayor proyeccion sobre el eje, y desde ese anclaje se construye un rectangulo orientado de DIMENSIONES[region] en la direccion del eje
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
    proj_eje_grid = dy * eje[0] + dx * eje[1]
    proj_perp_grid = dy * perp[0] + dx * perp[1]
    mascara_proy = (np.abs(proj_eje_grid) <= largo_px / 2.0) & \
                   (np.abs(proj_perp_grid) <= ancho_px / 2.0)

    info = {
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


def generar_mascara_inpainting(npz_path):
    #pipeline completo para una imagen: carga predicciones, identifica mutiladas con el filtro de descendiente, proyecta por simetria primero y por fallback anatomico despues, resta el cuerpo presente y dilata
    body_parts, bbox, score = cargar_predicciones(npz_path)
    H, W = body_parts.shape
    x0, y0, w, h = bbox
    alto_referencia = max(int(h), int(w))

    mascaras = calcular_mascaras_por_region(body_parts)
    coberturas, pix_total = calcular_coberturas(mascaras)

    info = {
        "score": score,
        "bbox": [int(x0), int(y0), int(w), int(h)],
        "alto_referencia": alto_referencia,
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

    estructura = generate_binary_structure(2, 2)
    mascara_final = binary_dilation(union_pre, structure=estructura, iterations=DILATAR_PX)
    mascara_final = mascara_final & ~cuerpo_presente

    return (mascara_final.astype(np.uint8) * 255), "ok", info


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    log_path = Path("/home/pfc/cescuder/tfg/logs/compute_mask_densepose.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    npz_files = sorted(DIR_NPZ.glob("*.npz"))
    log.info(f"encontrados {len(npz_files)} archivos npz en {DIR_NPZ}")

    procesadas = 0
    errores = 0
    contadores_status = {}

    for npz_path in tqdm(npz_files, desc="generating masks"):
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

    log.info("mask generation completed")
    log.info(f"  procesadas: {procesadas}")
    for status, count in contadores_status.items():
        log.info(f"  status ({status}): {count}")
    log.info(f"  errores: {errores}")
    log.info(f"  output: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
