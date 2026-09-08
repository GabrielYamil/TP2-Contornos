"""Proyecto 1: deteccion y clasificacion de contornos con matchShapes."""

import cv2

from cargar_referencias import contornos_referencia
from vision import crear_mascara, contornos_validos


def nada(_):
    pass


def main():
    if not contornos_referencia:
        raise RuntimeError("No se pudieron cargar los contornos de referencia.")

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    cv2.namedWindow("Controles P1")
    cv2.createTrackbar("Umbral", "Controles P1", 127, 255, nada)
    cv2.createTrackbar("Kernel", "Controles P1", 1, 20, nada)
    cv2.createTrackbar("Area minima", "Controles P1", 500, 50000, nada)
    cv2.createTrackbar("Distancia maxima", "Controles P1", 10, 100, nada)

    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("No se pudo obtener la imagen de la camara.")
                break
            frame = cv2.flip(frame, 1)
            gris, binaria, mascara = crear_mascara(
                frame,
                cv2.getTrackbarPos("Umbral", "Controles P1"),
                cv2.getTrackbarPos("Kernel", "Controles P1"),
            )
            contornos = contornos_validos(
                mascara, cv2.getTrackbarPos("Area minima", "Controles P1")
            )
            distancia_maxima = cv2.getTrackbarPos(
                "Distancia maxima", "Controles P1"
            ) / 100.0

            for contorno in contornos:
                distancias = {
                    nombre: cv2.matchShapes(
                        contorno, referencia, cv2.CONTOURS_MATCH_I1, 0.0
                    )
                    for nombre, referencia in contornos_referencia.items()
                }
                nombre, distancia = min(distancias.items(), key=lambda item: item[1])
                reconocido = distancia <= distancia_maxima
                texto = f"{nombre} ({distancia:.3f})" if reconocido else "desconocido"
                color = (0, 255, 0) if reconocido else (0, 0, 255)
                x, y, ancho, alto = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, 2)
                cv2.putText(
                    frame,
                    texto,
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                f"Distancia maxima: {distancia_maxima:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Proyecto 1 - matchShapes", frame)
            cv2.imshow("Proyecto 1 - Gris", gris)
            cv2.imshow("Proyecto 1 - Binaria", binaria)
            cv2.imshow("Proyecto 1 - Morfologia", mascara)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
