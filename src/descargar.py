"""Descarga fotos de animales desde Wikipedia (en español).

El Excel viene de Wikipedia, así que el nombre de cada animal (columna Entity)
es el título de su artículo. Le pedimos a Wikipedia la foto principal de cada uno.
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

from datos import CARPETA_IMAGENES, RUTA_MANIFIESTO, SEMILLA

API = "https://es.wikipedia.org/w/api.php"
# Wikipedia pide identificarse al descargar de forma automática
CABECERAS = {"User-Agent": "clasificador-fauna/1.0 (proyecto universitario de deep learning)"}


def buscar_urls(titulos):
    """Pregunta a Wikipedia la foto principal de hasta 50 artículos a la vez.

    Devuelve un diccionario {título: url_de_la_foto} solo con los que tienen foto.
    """
    respuesta = requests.get(API, headers=CABECERAS, timeout=30, params={
        "action": "query", "format": "json", "prop": "pageimages", "piprop": "thumbnail",
        "pithumbsize": 320, "pilimit": 50, "redirects": 1, "titles": "|".join(titulos),
    })
    datos = respuesta.json().get("query", {})

    # Wikipedia puede cambiar un título (mayúsculas) o redirigirlo a otro artículo.
    # Seguimos esos cambios para saber qué foto corresponde a cada título original.
    destino = {t: t for t in titulos}
    for cambios in (datos.get("normalized", []), datos.get("redirects", [])):
        mapa = {c["from"]: c["to"] for c in cambios}
        destino = {t: mapa.get(d, d) for t, d in destino.items()}

    fotos = {p["title"]: p["thumbnail"]["source"]
             for p in datos.get("pages", {}).values() if "thumbnail" in p}
    return {t: fotos[d] for t, d in destino.items() if d in fotos}


def descargar_foto(url, destino):
    """Descarga una foto y la guarda como JPG. Devuelve True si salió bien."""
    for intento in range(3):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=30)
            if r.status_code == 429:          # demasiadas peticiones: esperar y reintentar
                time.sleep(3 * (intento + 1))
                continue
            r.raise_for_status()
            imagen = Image.open(io.BytesIO(r.content)).convert("RGB")
            if min(imagen.size) < 100:        # muy pequeña: la descartamos
                return False
            imagen.save(destino, "JPEG", quality=90)
            return True
        except Exception:
            time.sleep(1)
    return False


def descargar_imagenes(tabla, por_grupo=500, carpeta=CARPETA_IMAGENES, ruta_manifiesto=RUTA_MANIFIESTO):
    """Descarga hasta `por_grupo` fotos de cada grupo animal.

    tabla: resultado de tabla_entidades(). Guarda las fotos en carpeta/<grupo>/
    y crea un archivo manifiesto.csv con el animal, grupo, archivo y url de cada foto.
    """
    filas = []
    for grupo, animales in tabla.groupby("grupo"):
        animales = animales.sample(frac=1, random_state=SEMILLA)  # orden al azar
        (carpeta / grupo).mkdir(parents=True, exist_ok=True)
        urls_vistas = set()   # para no repetir la misma foto (ej. dos especies del mismo género)
        guardadas = 0

        # Pedimos las fotos de 50 en 50 animales hasta juntar las necesarias
        for inicio in range(0, len(animales), 50):
            if guardadas >= por_grupo:
                break
            lote = animales.iloc[inicio:inicio + 50]
            nombres = [n for n in lote["Entity"] if not any(c in n for c in "|#<>[]{}")]
            urls = buscar_urls(nombres)

            tareas = []
            for posicion, (_, animal) in enumerate(lote.iterrows()):
                url = urls.get(animal["Entity"])
                # Solo fotos JPG (los PNG y SVG suelen ser mapas o dibujos)
                if not url or url in urls_vistas or not url.lower().endswith((".jpg", ".jpeg")):
                    continue
                urls_vistas.add(url)
                archivo = f"{grupo}/{grupo}_{inicio + posicion:05d}.jpg"  # nombre único
                tareas.append((animal, url, archivo))
            tareas = tareas[:por_grupo - guardadas]   # no descargar más de las que faltan

            with ThreadPoolExecutor(max_workers=4) as pool:
                resultados = list(pool.map(lambda t: descargar_foto(t[1], carpeta / t[2]), tareas))

            for (animal, url, archivo), ok in zip(tareas, resultados):
                if ok and guardadas < por_grupo:
                    filas.append({"entidad": animal["Entity"], "pais": animal["Country"],
                                  "grupo": grupo, "archivo": archivo, "url": url})
                    guardadas += 1
            time.sleep(0.2)

        print(f"{grupo:<10} {guardadas} fotos (de {len(animales)} animales)")

    manifiesto = pd.DataFrame(filas)
    manifiesto.to_csv(ruta_manifiesto, index=False)
    return manifiesto
