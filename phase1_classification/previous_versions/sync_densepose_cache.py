#sync_densepose_cache.py
#sincroniza el densepose_cache con la nueva distribucion de imagenes en dataset_classificado
#mueve los .npz a la categoria correcta y elimina los huerfanos cuya imagen ya no existe
#uso: python sync_densepose_cache.py

import shutil
from pathlib import Path

#rutas en el cluster
DATASET_DIR = Path("/home/pfc/cescuder/tfg/dataset_classificado")
CACHE_DIR = Path("/home/pfc/cescuder/tfg/densepose_cache")

#las 4 categorias del pipeline
CATEGORIAS = ["whole_body", "broken_body", "head_only", "no_human"]


def construir_mapa_imagenes():
    #devuelve un dict que mapea el nombre del fichero (sin extension) a la categoria actual
    imagen_a_categoria = {}
    for cat in CATEGORIAS:
        cat_dir = DATASET_DIR / cat
        if not cat_dir.exists():
            print(f"  aviso: la carpeta {cat_dir} no existe, se ignora")
            continue
        for img_path in cat_dir.iterdir():
            if img_path.is_file():
                stem = img_path.stem
                imagen_a_categoria[stem] = cat
    return imagen_a_categoria


def main():
    print("[1/3] mapeando imagenes a categorias actuales...")
    imagen_a_categoria = construir_mapa_imagenes()
    print(f"  total imagenes en el dataset: {len(imagen_a_categoria)}")
    for cat in CATEGORIAS:
        n = sum(1 for c in imagen_a_categoria.values() if c == cat)
        print(f"    {cat}: {n}")

    print("\n[2/3] recorriendo el cache de .npz...")
    todos_npz = list(CACHE_DIR.rglob("*.npz"))
    print(f"  total .npz encontrados: {len(todos_npz)}")

    movidos = 0
    borrados = 0
    sin_cambio = 0

    for npz_path in todos_npz:
        stem = npz_path.stem
        cat_destino = imagen_a_categoria.get(stem)

        if cat_destino is None:
            #imagen ya no existe en ninguna carpeta, .npz huerfano
            npz_path.unlink()
            borrados += 1
            continue

        destino_dir = CACHE_DIR / cat_destino
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino_path = destino_dir / npz_path.name

        if npz_path.resolve() == destino_path.resolve():
            #ya esta donde toca, no se hace nada
            sin_cambio += 1
        else:
            #mover a la categoria correcta
            shutil.move(str(npz_path), str(destino_path))
            movidos += 1

    print("\n[3/3] resumen final:")
    print(f"  movidos a otra categoria:        {movidos}")
    print(f"  borrados por huerfanos:          {borrados}")
    print(f"  sin cambios (ya en su sitio):    {sin_cambio}")
    print(f"  total .npz tras sync:            {movidos + sin_cambio}")

    #conteo final por carpeta
    print("\n  distribucion final del cache:")
    for cat in CATEGORIAS:
        cat_dir = CACHE_DIR / cat
        if cat_dir.exists():
            n = len(list(cat_dir.glob("*.npz")))
            print(f"    {cat}: {n} .npz")


if __name__ == "__main__":
    main()
