"""Proyecto 2: clasificador de contornos usando el modelo entrenado."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from joblib import load

from vision import crear_mascara, contornos_validos, invariantes_hu, transformar_hu


def nada(_):
    """Callback requerido por las barras deslizantes de OpenCV."""
    pass


def main():
    # El modelo y la cámara pueden cambiarse mediante argumentos opcionales.
    parser = argparse.ArgumentParser(description="Clasificador ML de contornos")
    parser.add_argument("--modelo", default="modelos/clasificador_hu.joblib")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    ruta = Path(args.modelo)
    if not ruta.exists():
        raise FileNotFoundError("No existe el modelo. Ejecute primero: python entrenar.py")
    # Recupera el árbol ya entrenado y el diccionario de nombres.
    modelo = load(ruta)
    clasificador = modelo["clasificador"]
    etiquetas = modelo["etiquetas"]

    # Abre la webcam elegida; normalmente la cámara principal tiene índice 0.
    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    # Los controles ajustan la segmentación antes de realizar la predicción.
    cv2.namedWindow("Controles ML")
    cv2.createTrackbar("Umbral", "Controles ML", 127, 255, nada)
    cv2.createTrackbar("Kernel", "Controles ML", 1, 20, nada)
    cv2.createTrackbar("Area minima", "Controles ML", 500, 50000, nada)
    cv2.createTrackbar("Invertir", "Controles ML", 0, 1, nada)

    try:
        while True:
            # Captura un fotograma y lo refleja para obtener una vista tipo espejo.
            ok, frame = camara.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            # La máscara debe contener figuras blancas sobre un fondo negro.
            _, _, mascara = crear_mascara(
                frame,
                cv2.getTrackbarPos("Umbral", "Controles ML"),
                cv2.getTrackbarPos("Kernel", "Controles ML"),
                bool(cv2.getTrackbarPos("Invertir", "Controles ML")),
            )
            # Los contornos pequeños y los que tocan el borde se descartan.
            contornos = contornos_validos(
                mascara,
                cv2.getTrackbarPos("Area minima", "Controles ML"),
                descartar_borde=True,
            )
            for contorno in contornos:
                # Calcula exactamente las mismas características usadas al entrenar.
                muestra = transformar_hu(invariantes_hu(contorno)).reshape(1, -1)
                # predict devuelve la clase y predict_proba la confianza de la hoja.
                etiqueta = int(clasificador.predict(muestra)[0])
                confianza = float(np.max(clasificador.predict_proba(muestra)[0]))
                nombre = etiquetas.get(etiqueta, f"etiqueta {etiqueta}")
                # Anota la figura detectada sobre el fotograma original.
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
            # Muestra tanto la clasificación final como la máscara procesada.
            cv2.imshow("Proyecto 2 - Clasificador ML", frame)
            cv2.imshow("Proyecto 2 - Mascara", mascara)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        # Este bloque también se ejecuta si ocurre un error dentro del bucle.
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
