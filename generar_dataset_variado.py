"""Genera un dataset experimental con siluetas menos regulares."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from etiquetas import ETIQUETAS
from vision import invariantes_hu

ALTO = 480
ANCHO = 480
ENCABEZADO = [f"hu{i}" for i in range(1, 8)] + ["etiqueta"]


def puntos_circulo(centro, radio, rng):
    """Crea una circunferencia levemente irregular y elíptica."""
    cantidad = int(rng.integers(28, 65))
    angulos = np.linspace(0, 2 * np.pi, cantidad, endpoint=False)
    proporcion = float(rng.uniform(0.84, 1.16))
    fase = float(rng.uniform(0, 2 * np.pi))
    ondulacion = float(rng.uniform(0.01, 0.07))
    radios = radio * (
        1
        + ondulacion * np.sin(3 * angulos + fase)
        + rng.normal(0, 0.018, cantidad)
    )
    puntos = np.column_stack(
        (
            centro[0] + radios * np.cos(angulos),
            centro[1] + radios * proporcion * np.sin(angulos),
        )
    )
    return np.int32(puntos)


def puntos_poligono(centro, radio, lados, angulo, rng):
    """Crea cuadrados o triángulos con vértices ligeramente desplazados."""
    angulos = angulo + np.arange(lados) * (2 * np.pi / lados)
    angulos += rng.normal(0, 0.045 if lados == 4 else 0.065, lados)
    radios = radio * rng.uniform(0.84, 1.16, lados)
    puntos = np.column_stack(
        (
            centro[0] + radios * np.cos(angulos),
            centro[1] + radios * np.sin(angulos),
        )
    )
    return np.int32(puntos)


def deformar_perspectiva(imagen, rng):
    """Simula que la cámara no está completamente perpendicular."""
    margen = 28
    origen = np.float32(
        [[0, 0], [ANCHO - 1, 0], [ANCHO - 1, ALTO - 1], [0, ALTO - 1]]
    )
    destino = origen + rng.uniform(-margen, margen, origen.shape).astype(np.float32)
    matriz = cv2.getPerspectiveTransform(origen, destino)
    return cv2.warpPerspective(imagen, matriz, (ANCHO, ALTO), borderValue=0)


def simular_camara(silueta, rng):
    """Añade defectos habituales de foco, resolución, threshold y morfología."""
    fondo = float(rng.uniform(0, 45))
    figura = float(rng.uniform(175, 255))
    imagen = np.where(silueta > 0, figura, fondo).astype(np.float32)

    # Gradiente suave de iluminación y ruido del sensor.
    gradiente = np.linspace(
        float(rng.uniform(-25, 5)), float(rng.uniform(-5, 25)), ANCHO
    )
    if rng.random() < 0.5:
        gradiente = gradiente[::-1]
    imagen += gradiente[np.newaxis, :]
    imagen += rng.normal(0, float(rng.uniform(2, 12)), imagen.shape)
    imagen = np.uint8(np.clip(imagen, 0, 255))

    # Pérdida de detalle producida por distancia o baja resolución.
    escala = float(rng.uniform(0.40, 0.90))
    reducida = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    imagen = cv2.resize(reducida, (ANCHO, ALTO), interpolation=cv2.INTER_LINEAR)

    desenfoque = int(rng.choice([1, 3, 5, 7, 9]))
    if desenfoque > 1:
        imagen = cv2.GaussianBlur(imagen, (desenfoque, desenfoque), 0)

    # Otsu selecciona automáticamente un umbral para esta imagen sintética.
    _, mascara = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erosión o dilatación aleatoria para variar el borde segmentado.
    kernel_size = int(rng.choice([1, 3, 5, 7, 9]))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,) * 2)
        operacion = rng.choice([cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_ERODE, cv2.MORPH_DILATE])
        mascara = cv2.morphologyEx(mascara, int(operacion), kernel)
    return mascara


def crear_muestra(nombre, rng):
    """Dibuja una clase y devuelve la máscara que usaría el clasificador."""
    silueta = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    centro = (
        int(ANCHO / 2 + rng.integers(-70, 71)),
        int(ALTO / 2 + rng.integers(-70, 71)),
    )
    radio = int(rng.integers(75, 145))
    angulo = float(rng.uniform(0, 2 * np.pi))

    if nombre == "circulo":
        puntos = puntos_circulo(centro, radio, rng)
    elif nombre == "cuadrado":
        puntos = puntos_poligono(centro, radio * 1.20, 4, angulo, rng)
    elif nombre == "triangulo":
        puntos = puntos_poligono(centro, radio * 1.25, 3, angulo, rng)
    else:
        raise ValueError(f"Forma desconocida: {nombre}")

    cv2.fillPoly(silueta, [puntos], 255)
    silueta = deformar_perspectiva(silueta, rng)
    return simular_camara(silueta, rng)


def obtener_contorno(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise RuntimeError("Una muestra no produjo ningún contorno.")
    return max(contornos, key=cv2.contourArea)


def main():
    parser = argparse.ArgumentParser(description="Dataset experimental más variado")
    parser.add_argument("--cantidad", type=int, default=100, help="muestras por clase")
    parser.add_argument("--salida", default="data/dataset_variado.csv")
    parser.add_argument("--imagenes", default="data/imagenes_variadas")
    parser.add_argument("--semilla", type=int, default=2026)
    args = parser.parse_args()

    if args.cantidad < 2:
        raise ValueError("Se necesitan al menos dos muestras por clase.")

    rng = np.random.default_rng(args.semilla)
    salida = Path(args.salida)
    carpeta_imagenes = Path(args.imagenes)
    salida.parent.mkdir(parents=True, exist_ok=True)
    carpeta_imagenes.mkdir(parents=True, exist_ok=True)

    filas = []
    for etiqueta, nombre in ETIQUETAS.items():
        carpeta_clase = carpeta_imagenes / nombre
        carpeta_clase.mkdir(parents=True, exist_ok=True)
        for numero in range(1, args.cantidad + 1):
            mascara = crear_muestra(nombre, rng)
            ruta = carpeta_clase / f"{nombre}_{numero:03d}.png"
            if not cv2.imwrite(str(ruta), mascara):
                raise RuntimeError(f"No se pudo guardar {ruta}")
            hu = invariantes_hu(obtener_contorno(mascara))
            filas.append([*hu.tolist(), etiqueta])

    rng.shuffle(filas)
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(ENCABEZADO)
        escritor.writerows(filas)

    print(f"Dataset experimental: {salida}")
    print(f"Imágenes experimentales: {carpeta_imagenes}")
    print(f"Total: {len(filas)} muestras ({args.cantidad} por clase)")


if __name__ == "__main__":
    main()
