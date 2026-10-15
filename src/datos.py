"""Funciones para preparar los datos: etiquetas, división y carga de imágenes."""
import re
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------- Rutas y constantes ----------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_EXCEL = RAIZ / "data" / "dataset.xlsx"
CARPETA_IMAGENES = RAIZ / "data" / "imagenes"
RUTA_MANIFIESTO = RAIZ / "data" / "manifiesto.csv"

TAM_IMAGEN = 160   # todas las imágenes se ajustan a 160 x 160 píxeles
SEMILLA = 42

# ---------- Etiquetado a partir del Excel ----------
# La columna "Answer" de la pregunta fácil dice cosas como
# "una especie de anfibios de la familia..." o "ave paseriforme".
# Buscamos estas palabras clave para saber a qué grupo pertenece el animal.
PALABRAS_CLAVE = {
    "ave": [
        "ave", "aves", "paseriforme", "paseriformes", "apodiforme", "colibrí", "pájaro",
        "pájaros", "rapaz", "loro", "loros", "gavilán", "águila", "búho", "tucán", "paloma",
        "pato", "tangara", "atrapamoscas", "hornero", "carpintero", "halcón", "garza",
        "lechuza", "playero", "chotacabras", "tinamú", "trogón", "cotinga", "psitaciforme",
        "falconiforme", "anseriforme", "columbiforme", "galliforme", "piciforme", "strigiforme",
        "caprimulgiforme", "charadriiforme", "pelecaniforme", "suliforme", "accipitriforme",
        "cuculiforme", "gruiforme", "chorlito", "mosquerito", "pinzón", "tordo", "canario",
        "cardenal", "jilguero", "gaviota", "pingüino", "pelícano", "flamenco", "cóndor",
        "zorzal", "golondrina", "tirano", "semillero", "hormiguero", "trepatroncos", "cacique",
        "oropéndola", "reinita", "ruiseñor", "codorniz", "perdiz", "pava", "guacamayo",
        "cotorra", "periquito", "urraca", "mirlo", "tangará", "tiránido", "apodidae",
    ],
    "mamífero": [
        "mamífero", "mamíferos", "roedor", "roedores", "murciélago", "murciélagos", "primate",
        "primates", "marsupial", "marsupiales", "felino", "cánido", "ardilla", "zarigüeya",
        "ratón", "delfín", "ballena", "mono", "monos", "venado", "tapir", "comadreja",
        "carnívoro", "cetáceo", "armadillo", "perezoso", "jaguar", "puma", "zorro", "nutria",
        "quiróptero", "quirópteros", "musaraña", "cérvido", "tucu-tucu",
    ],
    "reptil": [
        "reptil", "reptiles", "serpiente", "serpientes", "lagarto", "lagartos", "lagartija",
        "lagartijas", "tortuga", "tortugas", "culebra", "culebras", "iguana", "iguanas",
        "iguanios", "saurio", "escamoso", "escamosos", "caimán", "cocodrilo", "gecko", "geco",
        "víbora", "boa", "anolis", "camaleón", "serpentes", "ofidio", "dipsádido", "colúbrido",
        "gecónido", "teiido", "gimnoftálmido", "anguido", "crotálido", "elápido",
        "tropidúrido", "sceloporus",
    ],
    "anfibio": [
        "anfibio", "anfibios", "anuro", "anuros", "anura", "rana", "ranas", "sapo", "sapos",
        "salamandra", "salamandras", "cecilia", "cecilias", "caudado", "caudados", "ranita",
        "ránido", "ránidos", "leptodactílido", "hílido", "hílidos", "bufónido", "bufónidos",
        "dendrobátido", "dendrobátidos", "craugastórido", "strabomantidae", "centrolénido",
        "plethodontidae", "pletodóntido", "pletodóntidos", "ambystomatidae",
    ],
    "pez": [
        "pez", "peces", "bagre", "bagres", "tiburón", "tiburones", "raya", "rayas",
        "characiforme", "siluriforme", "cíclido", "cíclidos", "tetra", "piraña", "carpa",
        "pargo", "mojarra", "anguila", "perciforme", "trucha", "sardina", "siluro", "rivulus",
        "poecílido", "poecílidos", "loricárido", "loricáridos", "gimnotiforme", "góbido",
        "góbidos", "pimelodidae", "pimelódido", "serrasalmidae", "characidae", "carácido",
        "carácidos", "astyanax", "killifish", "trichomycteridae", "callichthyidae",
    ],
    "insecto": [
        "insecto", "insectos", "escarabajo", "escarabajos", "polilla", "polillas", "mariposa",
        "mariposas", "abeja", "abejas", "avispa", "avispas", "hormiga", "hormigas", "mosca",
        "moscas", "libélula", "libélulas", "saltamontes", "coleóptero", "coleópteros",
        "lepidóptero", "lepidópteros", "díptero", "himenóptero", "chinche", "chinches",
        "cucaracha", "cucarachas", "grillo", "grillos", "mosquito", "mosquitos", "termita",
        "cigarra", "mantis", "hemíptero", "ortóptero", "longicornio", "cerambícido",
        "curculiónido", "crisomélido", "gorgojo", "noctuido", "geométrido", "pirálido",
        "tortrícido", "arctiino", "hesperiido", "ninfálido", "nymphalidae", "sphingidae",
        "saturniidae", "esfíngido", "cicadélido", "membrácido", "reduviidae", "pentatómido",
        "tábano", "abejorro", "carábido", "escarabeido", "scarabaeidae", "tenebriónido",
        "bupréstido", "coccinélido", "mariquita", "efímera", "tricóptero", "plecóptero",
        "dermáptero", "fásmido", "oruga",
    ],
}
GRUPOS = list(PALABRAS_CLAVE)  # ['ave', 'mamífero', 'reptil', 'anfibio', 'pez', 'insecto']

# Un patrón por grupo: busca cualquiera de sus palabras como palabra completa
PATRONES = {
    grupo: re.compile(r"\b(" + "|".join(map(re.escape, palabras)) + r")\b")
    for grupo, palabras in PALABRAS_CLAVE.items()
}


def etiquetar_grupo(respuesta):
    """Devuelve el grupo animal de una respuesta, o None si no se reconoce.

    Solo se mira el comienzo de la respuesta (70 caracteres) y gana la palabra
    que aparece primero: "una especie de ave ... que come insectos" -> "ave".
    """
    comienzo = str(respuesta).lower()[:70]
    posiciones = {}
    for grupo, patron in PATRONES.items():
        coincidencia = patron.search(comienzo)
        if coincidencia:
            posiciones[grupo] = coincidencia.start()
    if not posiciones:
        return None
    return min(posiciones, key=posiciones.get)


def tabla_entidades(ruta=RUTA_EXCEL):
    """Lee el Excel y devuelve una fila por animal con su grupo.

    Columnas: Entity (nombre), Country (país) y grupo (ave, pez, ...).
    """
    df = pd.read_excel(ruta)

    # La pregunta FÁCIL de fauna es "¿qué tipo de especie es X?", y su respuesta
    # nos dice el grupo. Las otras preguntas hablan del hábitat o la dieta.
    df = df[(df["Category"] == "fauna") & (df["Difficulty"] == "FÁCIL")].copy()
    df["grupo"] = df["Answer"].map(etiquetar_grupo)
    df = df.dropna(subset=["grupo"])

    # Una sola fila por animal (y sin entidades con etiquetas contradictorias)
    por_entidad = df.groupby("Entity").agg(grupo=("grupo", "first"), n_grupos=("grupo", "nunique"),
                                           Country=("Country", "first")).reset_index()
    por_entidad = por_entidad[por_entidad["n_grupos"] == 1]
    return por_entidad[["Entity", "Country", "grupo"]].reset_index(drop=True)


# ---------- División y carga de imágenes ----------
def cargar_manifiesto(ruta=RUTA_MANIFIESTO):
    """Lee la lista de imágenes descargadas y agrega la ruta completa de cada archivo."""
    df = pd.read_csv(ruta)
    df["ruta"] = df["archivo"].map(lambda a: str(CARPETA_IMAGENES / a))
    return df


def dividir_datos(df):
    """Divide en entrenamiento (70%), validación (15%) y prueba (15%).

    Se mantiene la misma proporción de cada grupo en los tres conjuntos (stratify).
    """
    train, resto = train_test_split(df, test_size=0.30, stratify=df["grupo"], random_state=SEMILLA)
    val, test = train_test_split(resto, test_size=0.50, stratify=resto["grupo"], random_state=SEMILLA)
    return train, val, test


def _leer_imagen(ruta, etiqueta):
    """Lee un archivo JPG y lo deja en 160 x 160 x 3."""
    imagen = tf.io.read_file(ruta)
    imagen = tf.io.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, (TAM_IMAGEN, TAM_IMAGEN))
    return imagen, etiqueta


def crear_dataset(rutas, etiquetas, entrenamiento=False, lote=32):
    """Crea un dataset de TensorFlow listo para entrenar o evaluar.

    rutas: lista de archivos. etiquetas: lista de números (0, 1, 2...).
    """
    ds = tf.data.Dataset.from_tensor_slices((list(rutas), list(etiquetas)))
    if entrenamiento:
        ds = ds.shuffle(len(rutas), seed=SEMILLA)   # mezclar solo al entrenar
    ds = ds.map(_leer_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(lote).prefetch(tf.data.AUTOTUNE)
