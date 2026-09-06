import cv2
import os


def obtener_contorno(ruta):

    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta}")
        return None


    _, binaria = cv2.threshold(
        imagen,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"No se encontraron contornos en la imagen: {ruta}")
        return None

    contorno = max(
        contours,
        key=cv2.contourArea
    )

    return contorno


referencias = {
    "triangulo": "referencias/triangulo.png",
    "cuadrado": "referencias/cuadrado.png",
    "circulo": "referencias/circulo.png"
}

contornos_referencia = {}

for nombre, ruta in referencias.items():

    contorno = obtener_contorno(ruta)

    if contorno is not None:

        contornos_referencia[nombre] = contorno

        area = cv2.contourArea(contorno)

        print(
            f"{nombre} contorno encontrado "
            f"(area = {area:.2f})"
        )