# 🐾 Clasificador de grupos de fauna con una CNN

Proyecto de Deep Learning con **Redes Neuronales Convolucionales (CNN)** para clasificar imágenes:
dada la foto de un animal de Latinoamérica, el modelo dice si es un **ave, mamífero, reptil, anfibio, pez o insecto**.
Incluye una aplicación en **Streamlit** donde se sube una foto y se muestra la clase predicha con su nivel de confianza.

## ¿Cómo se usa el dataset (Excel)?

El archivo `data/dataset.xlsx` tiene 105.000 preguntas y respuestas de cultura latinoamericana. **No trae imágenes**,
así que se usa para **conseguir y etiquetar las fotos**:

1. Se toman los animales de la categoría `fauna` (columna `Category`).
2. La respuesta a «¿qué tipo de especie es…?» (dificultad FÁCIL) indica el grupo. Por ejemplo,
   *«una especie de anfibios de la familia…»* → `anfibio`. Se detecta con palabras clave (`src/datos.py`).
3. El nombre del animal (`Entity`) es el título de su artículo en Wikipedia en español. Con la API de Wikipedia se
   descarga su foto principal (`src/descargar.py`).

Del Excel salen unos **4.000 animales** con grupo claro (anfibios, insectos, aves, peces, mamíferos y reptiles).
Se descargan hasta **500 fotos por grupo** (no todos los animales tienen foto).

## Estructura del repositorio

```
clasificador_fauna/
├── notebook_colab.ipynb        # Notebook de experimentación (hecho para Google Colab)
├── src/
│   ├── datos.py                # Etiquetado desde el Excel, división y carga de imágenes
│   ├── descargar.py            # Descarga de fotos desde Wikipedia
│   ├── modelo.py               # CNN desde cero y MobileNetV2 (transferencia de aprendizaje)
│   ├── entrenar.py             # Entrena y guarda el modelo desde la terminal
│   └── predecir.py             # Carga el modelo y predice una foto
├── app.py                      # Aplicación Streamlit
├── data/
│   ├── dataset.xlsx            # Dataset original
│   ├── manifiesto.csv          # (se crea al descargar) animal, grupo, archivo y URL de cada foto
│   └── imagenes/               # (se crea al descargar) fotos por grupo
├── modelo/                     # modelo.keras, clases.json y metricas.json (se crean al entrenar)
├── imagenes/                   # Gráficas generadas por el notebook
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Paso 1: entrenar en Google Colab (notebook)

1. Subir este repositorio a GitHub (o ten a mano el `.zip` del proyecto).
2. Abrir `notebook_colab.ipynb` en [Google Colab](https://colab.research.google.com)
   (*Archivo → Abrir cuaderno → GitHub*, o súbelo directamente).
3. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
4. En la primera celda cambiar `REPO_URL` por la URL de tu repositorio. Si se dejas igual, el notebook pedirá
   subir el `.zip` del proyecto.
5. Ejecutar todo: *Entorno de ejecución → Ejecutar todo*. La descarga de fotos tarda unos 5–15 minutos y el
   entrenamiento unos minutos más.
6. Al final, la última celda descarga `modelo.zip`, `imagenes.zip`, `data_imagenes.zip` y `manifiesto.csv`.
   Descomprimirlos en las carpetas del proyecto (`modelo/`, `imagenes/`, `data/imagenes/`, `data/manifiesto.csv`).
7. Para que el notebook quede con sus resultados en el repositorio: *Archivo → Guardar una copia en GitHub*,
   o descárgalo con *Archivo → Descargar → .ipynb* y reemplaza `notebook_colab.ipynb`.

### Paso 2: abrir la aplicación Streamlit (en tu computador)

Requiere Python 3.10, 3.11 o 3.12 (TensorFlow aún no soporta versiones más nuevas en todos los casos).

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app carga `modelo/modelo.keras`, por lo que debe existir (paso 1). Se abre en `http://localhost:8501`.

> Si TensorFlow no se deja instalar con `tensorflow` en Windows, revisa la versión de Python con `python --version`.

### Opcional: entrenar desde la terminal

```bash
python src/entrenar.py             # MobileNetV2 (descarga las fotos si no existen)
python src/entrenar.py cnn         # CNN desde cero
python src/predecir.py foto.jpg    # predecir una foto
```

## Metodología

| Etapa | Qué se hace |
|---|---|
| Exploración | Conteo por grupo, ejemplos de fotos y distribución de tamaños |
| Preprocesamiento | Fotos a 160 × 160 píxeles; normalización dentro del modelo (capa `Rescaling`) |
| División | 70 % entrenamiento, 15 % validación, 15 % prueba, manteniendo la proporción de cada grupo |
| Data Augmentation | Volteo horizontal, rotación, zoom y contraste (solo al entrenar) |
| Desbalance | Pesos por clase en el entrenamiento y métrica F1 macro |
| Modelo A | CNN desde cero: 4 bloques `Conv2D + MaxPooling`, `GlobalAveragePooling`, `Dense` y `Dropout` |
| Modelo B | MobileNetV2 preentrenada en ImageNet (congelada) + capa final nueva |
| Entrenamiento | Adam, *early stopping* sobre la pérdida de validación |
| Evaluación | Accuracy, F1 macro, reporte por clase y matriz de confusión sobre el conjunto de prueba |
| Selección | Se guarda el modelo con mejor F1 en validación |

## Resultados

Los resultados se generan al ejecutar el notebook y quedan en:
- `modelo/metricas.json`: accuracy y F1 macro del modelo guardado sobre el conjunto de prueba.
- `imagenes/`: curvas de entrenamiento, matriz de confusión y 15 predicciones de prueba.
- La sección 8 del notebook compara la CNN desde cero con MobileNetV2.

## Limitaciones

- Las etiquetas salen de palabras clave en las respuestas del Excel, así que puede haber algún error.
- Las fotos de Wikipedia son muy variadas (animal pequeño, ejemplares disecados, ilustraciones…).
- Los grupos están desbalanceados (hay menos reptiles y mamíferos).
- No hay clase "otro": con la foto de algo que no sea un animal, el modelo igualmente elegirá un grupo.

## Licencias de las imágenes

Las fotos provienen de Wikipedia/Wikimedia Commons y cada una tiene su propia licencia (en su mayoría Creative Commons).
Se usan solo con fines académicos. El archivo `data/manifiesto.csv` guarda la URL de cada foto para poder
atribuirla.

| Modelo | Accuracy (prueba) | F1 macro (prueba) |
|---|---|---|
| CNN desde cero | 0.404 | 0.372 |
| **MobileNetV2 (transferencia de aprendizaje)** | **0.877** | **0.840** |

Se descargaron 1.563 fotos (70 % entrenamiento, 15 % validación, 15 % prueba = 235 fotos).
MobileNetV2 fue el mejor modelo y es el que usa la app. Los grupos con mejor resultado
fueron ave (F1 0.97) y mamífero (0.91); el más difícil fue reptil (0.65), que se confunde
sobre todo con anfibio. La CNN desde cero rinde mucho peor porque hay pocas fotos de
entrenamiento, lo que muestra la ventaja de la transferencia de aprendizaje.


# 🐾 Clasificador de grupos de fauna con una CNN

Proyecto de Deep Learning con **Redes Neuronales Convolucionales (CNN)** para clasificar imágenes:
dada la foto de un animal de Latinoamérica, el modelo dice si es un **ave, mamífero, reptil, anfibio, pez o insecto**.
Incluye una aplicación en **Streamlit** donde se sube una foto y se muestra la clase predicha con su nivel de confianza.

**Resultado:** MobileNetV2 (transferencia de aprendizaje) alcanza **87,7 % de accuracy y 0,84 de F1 macro** en el
conjunto de prueba.

## Prueba rápida de la aplicación (modelo ya entrenado incluido)

El repositorio ya trae el modelo entrenado en `modelo/`, así que no hace falta entrenar nada para probar la app.
Requiere **Python 3.10, 3.11 o 3.12** y Git.

```bash
git clone https://github.com/SebastianCely/Red-Neuronal-Convolucional-para-detectar-que-tipo-de-animal-es-el-que-aparece-en-la-foto-adjuntada.git
cd Red-Neuronal-Convolucional-para-detectar-que-tipo-de-animal-es-el-que-aparece-en-la-foto-adjuntada

python -m venv .venv
source .venv/bin/activate          # En Windows (PowerShell): .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501` (la primera vez tarda unos segundos en cargar TensorFlow y el modelo).
Para probarla, sube la foto de un animal (JPG, PNG o WEBP): verás la clase predicha, la confianza y la probabilidad
de cada uno de los 6 grupos.

## ¿Cómo se usa el dataset (Excel)?

El archivo `data/dataset.xlsx` tiene 105.000 preguntas y respuestas de cultura latinoamericana. **No trae imágenes**,
así que se usa para **conseguir y etiquetar las fotos**:

1. Se toman los animales de la categoría `fauna` (columna `Category`).
2. La respuesta a «¿qué tipo de especie es…?» (dificultad FÁCIL) indica el grupo. Por ejemplo,
   *«una especie de anfibios de la familia…»* → `anfibio`. Se detecta con palabras clave (`src/datos.py`).
3. El nombre del animal (`Entity`) es el título de su artículo en Wikipedia en español. Con la API de Wikipedia se
   descarga su foto principal (`src/descargar.py`).

Del Excel salen unos **4.000 animales** con grupo claro. Como no todos tienen foto en Wikipedia, se obtuvieron
**1.563 fotos**: ave 500, anfibio 373, mamífero 274, pez 157, insecto 142 y reptil 117.

## Estructura del repositorio

```
├── notebook_colab.ipynb        # Notebook de experimentación (hecho en Google Colab, con resultados)
├── src/
│   ├── datos.py                # Etiquetado desde el Excel, división y carga de imágenes
│   ├── descargar.py            # Descarga de fotos desde Wikipedia
│   ├── modelo.py               # CNN desde cero y MobileNetV2 (transferencia de aprendizaje)
│   ├── entrenar.py             # Entrena y guarda el modelo desde la terminal
│   └── predecir.py             # Carga el modelo y predice una foto
├── app.py                      # Aplicación Streamlit
├── data/dataset.xlsx           # Dataset original
├── modelo/                     # modelo.keras, clases.json y metricas.json (modelo entrenado)
├── imagenes/                   # Gráficas generadas por el notebook
├── requirements.txt
└── README.md
```

## Cómo volver a entrenar el modelo

### Opción A: en Google Colab (la usada en este proyecto)

1. Abre `notebook_colab.ipynb` en [Google Colab](https://colab.research.google.com)
   (*Archivo → Abrir cuaderno → GitHub* y pega la URL de este repositorio).
2. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*.
3. En la primera celda, `REPO_URL` debe apuntar a este repositorio.
4. *Entorno de ejecución → Ejecutar todo*. La descarga de fotos tarda unos minutos y el entrenamiento otros pocos.

### Opción B: desde la terminal

```bash
python src/entrenar.py             # MobileNetV2 (descarga las fotos si no existen)
python src/entrenar.py cnn         # CNN desde cero
python src/predecir.py foto.jpg    # predecir una foto
```

> Las fotos se descargan de Wikipedia en el momento, así que al volver a entrenar el conjunto de fotos y los
> resultados pueden variar un poco.

## Metodología

| Etapa | Qué se hace |
|---|---|
| Exploración | Conteo por grupo, ejemplos de fotos y distribución de tamaños |
| Preprocesamiento | Fotos a 160 × 160 píxeles; normalización dentro del modelo (capa `Rescaling`) |
| División | 70 % entrenamiento, 15 % validación, 15 % prueba (235 fotos), manteniendo la proporción de cada grupo |
| Data Augmentation | Volteo horizontal, rotación, zoom y contraste (solo al entrenar) |
| Desbalance | Pesos por clase en el entrenamiento y métrica F1 macro |
| Modelo A | CNN desde cero: 4 bloques `Conv2D + MaxPooling`, `GlobalAveragePooling`, `Dense` y `Dropout` |
| Modelo B | MobileNetV2 preentrenada en ImageNet (congelada) + capa final nueva |
| Entrenamiento | Adam, *early stopping* sobre la pérdida de validación |
| Evaluación | Accuracy, F1 macro, reporte por clase y matriz de confusión sobre el conjunto de prueba |
| Selección | Se guarda el modelo con mejor F1 en validación (MobileNetV2) |

## Resultados (conjunto de prueba: 235 fotos)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| CNN desde cero | 0.404 | 0.372 |
| **MobileNetV2 (transferencia de aprendizaje)** | **0.877** | **0.840** |

Resultado por grupo con MobileNetV2:

| Grupo | Precisión | Recall | F1 | Fotos de prueba |
|---|---|---|---|---|
| ave | 0.97 | 0.97 | 0.97 | 75 |
| mamífero | 0.87 | 0.95 | 0.91 | 41 |
| reptil | 0.60 | 0.71 | 0.65 | 17 |
| anfibio | 0.88 | 0.77 | 0.82 | 56 |
| pez | 0.84 | 0.88 | 0.86 | 24 |
| insecto | 0.86 | 0.82 | 0.84 | 22 |

- MobileNetV2 fue el mejor modelo y es el que usa la app.
- La **CNN desde cero rinde mucho peor** porque hay pocas fotos de entrenamiento (unas 1.100); la transferencia de
  aprendizaje aprovecha lo que MobileNetV2 ya aprendió con millones de fotos.
- La confusión más frecuente es **anfibio ↔ reptil** (8 anfibios clasificados como reptil y 4 al revés).
- Las curvas de entrenamiento, la matriz de confusión y las 15 predicciones de prueba están en el notebook
  y en la carpeta `imagenes/`.

## Limitaciones

- El conjunto de prueba es pequeño (235 fotos, solo 17 reptiles), así que los resultados por grupo son aproximados.
- Las etiquetas salen de palabras clave en las respuestas del Excel, así que puede haber algún error.
- Las fotos de Wikipedia son muy variadas (animal pequeño, ejemplares disecados, ilustraciones…).
- No hay clase "otro": con la foto de algo que no sea un animal, el modelo igualmente elegirá un grupo.

## Problemas comunes

- **`pip install` no encuentra TensorFlow:** revisa `python --version`. Usa Python 3.10, 3.11 o 3.12.
- **Error al cargar el modelo (`keras` / `deserialize`):** actualiza con `pip install --upgrade tensorflow keras`.
- **La app dice "No se encontró el modelo entrenado":** verifica que exista `modelo/modelo.keras`
  y que ejecutes `streamlit run app.py` desde la carpeta del proyecto.

## Licencias de las imágenes

Las fotos provienen de Wikipedia/Wikimedia Commons y cada una tiene su propia licencia (en su mayoría Creative Commons).
Se usan solo con fines académicos. El notebook registra la URL de cada foto descargada en `data/manifiesto.csv`
(archivo que se genera al ejecutar la descarga).

