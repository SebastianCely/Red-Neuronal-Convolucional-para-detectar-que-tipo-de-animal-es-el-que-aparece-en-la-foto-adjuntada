"""Descarga fotos de animales desde Wikipedia (en español).

El Excel viene de Wikipedia, así que el nombre de cada animal (columna Entity)
es el título de su artículo. Le pedimos a Wikipedia la foto principal de cada uno.
"""
import io
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

from datos import CARPETA_IMAGENES, RUTA_MANIFIESTO, SEMILLA

API = "https://es.wikipedia.org/w/api.php"
# Wikipedia pide identificarse (con un enlace de contacto) al descargar de forma automática.
# Si te sale un error 403, cambia el enlace por el de tu propio GitHub.
CONTACTO = "https://github.com/SebastianCely"
CABECERAS = {"User-Agent": f"ClasificadorFauna/1.0 (proyecto academico de deep learning; {CONTACTO}) python-requests"}


def pedir_json(params, intentos=5):
    """Consulta la API de Wikipedia. Si responde algo que no es JSON
    (por ejemplo error 429 "demasiadas peticiones" o 403), espera y reintenta."""
    for intento in range(intentos):
        respuesta = requests.get(API, headers=CABECERAS, params=params, timeout=30)
        if respuesta.status_code == 200:
            try:
                return respuesta.json()
            except ValueError:
                pass
        time.sleep(3 * (intento + 1))
    raise RuntimeError(f"Wikipedia no respondió bien (código {respuesta.status_code}). "
                       f"Respuesta: {respuesta.text[:200]}")


def buscar_urls(titulos):
    """Pregunta a Wikipedia la foto principal de hasta 50 artículos a la vez.

    Devuelve un diccionario {título: url_de_la_foto} solo con los que tienen foto.
    """
    datos = pedir_json({
        "action": "query", "format": "json", "prop": "pageimages", "piprop": "thumbnail",
        "pithumbsize": 330, "pilimit": 50, "redirects": 1, "titles": "|".join(titulos),
    }).get("query", {})

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
    """Descarga una foto y la guarda como JPG.

    Devuelve (True, "") si salió bien, o (False, motivo) si falló.
    """
    motivo = "desconocido"
    for intento in range(3):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=30)
            if r.status_code == 429:          # demasiadas peticiones: esperar y reintentar
                motivo = "error 429 (demasiadas peticiones)"
                time.sleep(3 * (intento + 1))
                continue
            r.raise_for_status()
            imagen = Image.open(io.BytesIO(r.content)).convert("RGB")
            if min(imagen.size) < 100:        # muy pequeña: la descartamos
                return False, "foto muy pequeña"
            imagen.save(destino, "JPEG", quality=90)
            return True, ""
        except Exception as error:
            motivo = f"{type(error).__name__}: {error}"[:90]
            time.sleep(1)
    return False, motivo


def descargar_imagenes(tabla, por_grupo=500, carpeta=CARPETA_IMAGENES, ruta_manifiesto=RUTA_MANIFIESTO):
    """Descarga hasta `por_grupo` fotos de cada grupo animal.

    tabla: resultado de tabla_entidades(). Guarda las fotos en carpeta/<grupo>/
    y crea un archivo manifiesto.csv con el animal, grupo, archivo y url de cada foto.
    """
    filas = []
    fallos = Counter()   # cuenta los motivos por los que fallan las descargas
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

            with ThreadPoolExecutor(max_workers=2) as pool:
                resultados = list(pool.map(lambda t: descargar_foto(t[1], carpeta / t[2]), tareas))

            for (animal, url, archivo), (ok, motivo) in zip(tareas, resultados):
                if not ok:
                    fallos[motivo] += 1
                if ok and guardadas < por_grupo:
                    filas.append({"entidad": animal["Entity"], "pais": animal["Country"],
                                  "grupo": grupo, "archivo": archivo, "url": url})
                    guardadas += 1
            time.sleep(0.2)

        print(f"{grupo:<10} {guardadas} fotos (de {len(animales)} animales)")

    if not filas:   # no se descargó nada: avisar en vez de guardar un manifiesto vacío
        raise RuntimeError(f"No se pudo descargar ninguna foto. Motivos de los fallos: {dict(fallos)}")
    if fallos:
        print("\nFotos que no se pudieron descargar:", dict(fallos))

    manifiesto = pd.DataFrame(filas)
    manifiesto.to_csv(ruta_manifiesto, index=False)
    return manifiesto
