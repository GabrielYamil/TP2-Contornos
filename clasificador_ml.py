"""Proyecto 2: clasificador de contornos usando el modelo entrenado."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from joblib import load

from vision import crear_mascara, contornos_validos, invariantes_hu


def nada(_):
    pass


def main():
    parser = argparse.ArgumentParser(description="Clasificador ML de contornos")
    parser.add_argument("--modelo", default="modelos/clasificador_hu.joblib")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    ruta = Path(args.modelo)
    if not ruta.exists():
        raise FileNotFoundError("No existe el modelo. Ejecute primero: python entrenar.py")
    modelo = load(ruta)
    clasificador = modelo["clasificador"]
    etiquetas = modelo["etiquetas"]

    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    cv2.namedWindow("Controles ML")
    cv2.createTrackbar("Umbral", "Controles ML", 127, 255, nada)
    cv2.createTrackbar("Kernel", "Controles ML", 1, 20, nada)
    cv2.createTrackbar("Area minima", "Controles ML", 500, 50000, nada)

    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            _, _, mascara = crear_mascara(
                frame,
                cv2.getTrackbarPos("Umbral", "Controles ML"),
                cv2.getTrackbarPos("Kernel", "Controles ML"),
            )
            contornos = contornos_validos(
                mascara, cv2.getTrackbarPos("Area minima", "Controles ML")
            )
            for contorno in contornos:
                muestra = invariantes_hu(contorno).reshape(1, -1)
                etiqueta = int(clasificador.predict(muestra)[0])
                confianza = float(np.max(clasificador.predict_proba(muestra)[0]))
                nombre = etiquetas.get(etiqueta, f"etiqueta {etiqueta}")
                x, y, ancho, alto = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{nombre} ({confianza:.0%})",
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Proyecto 2 - Clasificador ML", frame)
            cv2.imshow("Proyecto 2 - Mascara", mascara)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
