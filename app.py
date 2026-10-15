"""Aplicación Streamlit: sube la foto de un animal y el modelo dice a qué grupo pertenece.

Ejecutar con:
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Para poder importar el código de la carpeta src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from predecir import CARPETA_MODELO, cargar_modelo, modelo_existe, predecir

st.set_page_config(page_title="¿Qué animal es?", page_icon="🐾")

EMOJIS = {"ave": "🐦", "mamífero": "🐒", "reptil": "🦎", "anfibio": "🐸", "pez": "🐟", "insecto": "🦋"}


@st.cache_resource  # el modelo se carga una sola vez
def obtener_modelo():
    return cargar_modelo()


st.title("🐾 ¿Qué grupo de animal es?")
st.write(
    "Sube la foto de un animal y una red neuronal convolucional (CNN) dirá si es un "
    "**ave, mamífero, reptil, anfibio, pez o insecto**."
)

# Si todavía no se ha entrenado el modelo, avisamos y paramos
if not modelo_existe():
    st.error("No se encontró el modelo entrenado en la carpeta `modelo/`. "
             "Ejecuta primero el notebook (o `python src/entrenar.py`).")
    st.stop()

modelo, clases = obtener_modelo()

# Datos del modelo en la barra lateral
with st.sidebar:
    st.header("Sobre el modelo")
    ruta_metricas = CARPETA_MODELO / "metricas.json"
    if ruta_metricas.exists():
        metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
        st.write(f"Modelo: **{metricas['modelo']}**")
        st.write(f"Accuracy en prueba: **{metricas['accuracy']:.1%}**")
        st.write(f"F1 macro en prueba: **{metricas['f1_macro']:.2f}**")
    st.write("Clases: " + ", ".join(clases))

archivo = st.file_uploader("Elige una foto", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Tu foto", width="stretch")

    resultado = predecir(modelo, clases, imagen)
    mejor_clase, mejor_prob = resultado[0]

    st.success(f"{EMOJIS.get(mejor_clase, '')} Es un(a) **{mejor_clase}**  —  confianza: {mejor_prob:.1%}")

    tabla = pd.DataFrame(resultado, columns=["Clase", "Probabilidad"]).set_index("Clase")
    st.bar_chart(tabla)

    if mejor_prob < 0.6:
        st.info("El modelo no está muy seguro. Prueba con una foto donde el animal se vea completo y cercano.")
