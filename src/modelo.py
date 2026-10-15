"""Construcción de las redes neuronales: una CNN desde cero y una con MobileNetV2."""
from tensorflow import keras
from tensorflow.keras import layers

from datos import TAM_IMAGEN


def capa_aumento():
    """Data Augmentation: genera variaciones de cada foto al entrenar.

    Estas capas solo actúan durante el entrenamiento; al predecir no hacen nada.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),   # voltear izquierda-derecha
        layers.RandomRotation(0.1),        # girar un poco
        layers.RandomZoom(0.15),           # acercar o alejar
        layers.RandomContrast(0.2),        # cambiar el contraste
    ], name="aumento_de_datos")


def compilar(modelo, tasa=0.001):
    modelo.compile(
        optimizer=keras.optimizers.Adam(tasa),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def crear_cnn(n_clases):
    """CNN hecha desde cero: 4 bloques de convolución + pooling."""
    modelo = keras.Sequential([
        keras.Input(shape=(TAM_IMAGEN, TAM_IMAGEN, 3)),
        capa_aumento(),
        layers.Rescaling(1 / 255),                       # píxeles de 0-255 a 0-1

        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),                 # resume cada mapa en un número
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),                             # evita memorizar
        layers.Dense(n_clases, activation="softmax"),    # probabilidad por clase
    ], name="cnn_desde_cero")
    return compilar(modelo)


def crear_mobilenet(n_clases, pesos="imagenet"):
    """CNN con transferencia de aprendizaje.

    MobileNetV2 ya aprendió a "ver" con millones de fotos (ImageNet).
    Dejamos esa parte congelada y solo entrenamos la capa final.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(TAM_IMAGEN, TAM_IMAGEN, 3), include_top=False, weights=pesos
    )
    base.trainable = False   # congelada: no se modifica

    modelo = keras.Sequential([
        keras.Input(shape=(TAM_IMAGEN, TAM_IMAGEN, 3)),
        capa_aumento(),
        layers.Rescaling(1 / 127.5, offset=-1),          # píxeles de 0-255 a -1...1 (lo que espera MobileNetV2)
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation="softmax"),
    ], name="mobilenetv2")
    return compilar(modelo)
