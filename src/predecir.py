"""Carga el modelo entrenado y predice el grupo animal de una foto.

Uso desde la terminal:
    python src/predecir.py ruta/a/foto.jpg
"""
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from datos import RAIZ, TAM_IMAGEN

CARPETA_MODELO = RAIZ / "modelo"


def modelo_existe():
    return (CARPETA_MODELO / "modelo.keras").exists()


def cargar_modelo():
    """Devuelve el modelo entrenado y la lista de clases."""
    modelo = keras.models.load_model(CARPETA_MODELO / "modelo.keras")
    clases = json.loads((CARPETA_MODELO / "clases.json").read_text(encoding="utf-8"))
    return modelo, clases


def predecir(modelo, clases, imagen):
    """Devuelve [(clase, probabilidad), ...] de mayor a menor.

    imagen: un objeto PIL.Image (por ejemplo, la foto que sube el usuario).
    """
    imagen = imagen.convert("RGB")
    matriz = tf.image.resize(np.array(imagen), (TAM_IMAGEN, TAM_IMAGEN))  # igual que en el entrenamiento
    probabilidades = modelo.predict(np.expand_dims(matriz, 0), verbose=0)[0]
    resultado = list(zip(clases, probabilidades))
    return sorted(resultado, key=lambda par: par[1], reverse=True)


if __name__ == "__main__":
    modelo, clases = cargar_modelo()
    for clase, prob in predecir(modelo, clases, Image.open(sys.argv[1])):
        print(f"{clase:<10} {prob:.1%}")
