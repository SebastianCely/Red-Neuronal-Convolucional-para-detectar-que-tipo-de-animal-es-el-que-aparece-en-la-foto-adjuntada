"""Descarga las fotos (si hace falta), entrena un modelo y lo guarda en modelo/.

Uso:
    python src/entrenar.py              # MobileNetV2 (recomendado)
    python src/entrenar.py cnn          # CNN desde cero
"""
import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from datos import (GRUPOS, RAIZ, RUTA_MANIFIESTO, SEMILLA, cargar_manifiesto,
                   crear_dataset, dividir_datos, tabla_entidades)
from descargar import descargar_imagenes
from modelo import crear_cnn, crear_mobilenet

CARPETA_MODELO = RAIZ / "modelo"


def main(tipo="mobilenet"):
    keras.utils.set_random_seed(SEMILLA)

    # 1. Fotos: se descargan solo si todavía no existen
    if not RUTA_MANIFIESTO.exists():
        descargar_imagenes(tabla_entidades(), por_grupo=500)
    df = cargar_manifiesto()

    # 2. División en entrenamiento, validación y prueba
    train, val, test = dividir_datos(df)
    print(f"Entrenamiento: {len(train)} | Validación: {len(val)} | Prueba: {len(test)}")

    # 3. Etiquetas numéricas: cada grupo pasa a ser un número (0, 1, 2...)
    numero = {grupo: i for i, grupo in enumerate(GRUPOS)}
    y_train = train["grupo"].map(numero).to_numpy()
    y_val = val["grupo"].map(numero).to_numpy()
    y_test = test["grupo"].map(numero).to_numpy()

    ds_train = crear_dataset(train["ruta"], y_train, entrenamiento=True)
    ds_val = crear_dataset(val["ruta"], y_val)
    ds_test = crear_dataset(test["ruta"], y_test)

    # 4. Pesos: los grupos con menos fotos (ej. reptiles) pesan más
    pesos = compute_class_weight("balanced", classes=np.arange(len(GRUPOS)), y=y_train)
    pesos = dict(enumerate(pesos))

    # 5. Modelo y entrenamiento (se detiene solo si deja de mejorar)
    modelo = crear_mobilenet(len(GRUPOS)) if tipo == "mobilenet" else crear_cnn(len(GRUPOS))
    modelo.summary()
    parar_temprano = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    modelo.fit(ds_train, validation_data=ds_val, epochs=30,
               class_weight=pesos, callbacks=[parar_temprano])

    # 6. Evaluación con fotos que el modelo nunca vio
    pred = modelo.predict(ds_test).argmax(axis=1)
    accuracy = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro")
    print(f"\nAccuracy en prueba: {accuracy:.3f}")
    print(f"F1 macro en prueba: {f1_macro:.3f}")

    # 7. Guardar el modelo y la lista de clases
    CARPETA_MODELO.mkdir(exist_ok=True)
    modelo.save(CARPETA_MODELO / "modelo.keras")
    (CARPETA_MODELO / "clases.json").write_text(json.dumps(GRUPOS, ensure_ascii=False), encoding="utf-8")
    (CARPETA_MODELO / "metricas.json").write_text(json.dumps(
        {"modelo": tipo, "accuracy": accuracy, "f1_macro": f1_macro}, indent=2), encoding="utf-8")
    print("Modelo guardado en", CARPETA_MODELO)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenet")
