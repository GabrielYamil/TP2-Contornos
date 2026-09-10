"""Captura muestras de invariantes de Hu desde webcam y las guarda en CSV."""

import argparse
import csv
from pathlib import Path

import cv2

from etiquetas import ETIQUETAS
from vision import crear_mascara, contornos_validos, invariantes_hu

ENCABEZADO = [f"hu{i}" for i in range(1, 8)] + ["etiqueta"]


def nada(_):
    pass


def preparar_csv(ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not ruta.exists() or ruta.stat().st_size == 0:
        with ruta.open("w", newline="", encoding="utf-8") as archivo:
            csv.writer(archivo).writerow(ENCABEZADO)


def main():
    parser = argparse.ArgumentParser(description="Generador de descriptores Hu")
    parser.add_argument("--salida", default="data/dataset.csv")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()
    salida = Path(args.salida)
    preparar_csv(salida)
    etiqueta = 1
    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")
    cv2.namedWindow("Controles")
    cv2.createTrackbar("Umbral", "Controles", 127, 255, nada)
    cv2.createTrackbar("Kernel", "Controles", 1, 20, nada)
    cv2.createTrackbar("Area minima", "Controles", 500, 50000, nada)
    cv2.createTrackbar("Invertir", "Controles", 0, 1, nada)
    print("Teclas: 1-3 etiqueta; ESPACIO guarda el contorno mayor; ESC sale.")
    try:
        while True:
            ok, frame = camara.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            gris, binaria, mascara = crear_mascara(
                frame,
                cv2.getTrackbarPos("Umbral", "Controles"),
                cv2.getTrackbarPos("Kernel", "Controles"),
                bool(cv2.getTrackbarPos("Invertir", "Controles")),
            )
            contornos = contornos_validos(
                mascara,
                cv2.getTrackbarPos("Area minima", "Controles"),
                descartar_borde=True,
            )
            mayor = max(contornos, key=cv2.contourArea, default=None)
            cv2.putText(frame, f"Etiqueta {etiqueta}: {ETIQUETAS[etiqueta]} | Espacio: guardar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if mayor is not None: cv2.drawContours(frame, [mayor], -1, (0, 255, 0), 2)
            cv2.imshow("Generador de descriptores", frame)
            cv2.imshow("Mascara", mascara)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27: break
            if ord("1") <= tecla <= ord("3"): etiqueta = tecla - ord("0")
            elif tecla == ord(" ") and mayor is not None:
                hu = invariantes_hu(mayor)
                with salida.open("a", newline="", encoding="utf-8") as archivo:
                    csv.writer(archivo).writerow([*hu, etiqueta])
                print(f"Muestra guardada: {ETIQUETAS[etiqueta]} -> {hu}")
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
