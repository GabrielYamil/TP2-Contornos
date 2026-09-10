import cv2
import os


def obtener_contorno(ruta):
    """Carga una imagen de referencia y devuelve su contorno principal."""
    # Las referencias solo necesitan un canal porque se procesan por intensidad.
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta}")
        return None


    # Las figuras originales son negras; se invierten para obtener blanco sobre negro.
    _, binaria = cv2.threshold(
        imagen,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Solo interesan los contornos externos de la figura.
    contours, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"No se encontraron contornos en la imagen: {ruta}")
        return None

    # Si hubiera ruido, se conserva el contorno de mayor área.
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

# Este diccionario almacena los contornos que utilizará matchShapes.
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
