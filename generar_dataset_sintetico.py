"""Genera imágenes sintéticas y un dataset de invariantes de Hu."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from etiquetas import ETIQUETAS
from vision import invariantes_hu

ENCABEZADO = [f"hu{i}" for i in range(1, 8)] + ["etiqueta"]
ANCHO = 480
ALTO = 480


def punto_rotado(centro, radio, angulo):
    """Calcula un vértice a una distancia y un ángulo respecto del centro."""
    radianes = np.deg2rad(angulo)
    return (
        int(centro[0] + radio * np.cos(radianes)),
        int(centro[1] + radio * np.sin(radianes)),
    )


def dibujar_figura(nombre, rng):
    """Crea una silueta con variaciones aleatorias controladas."""
    # Cada muestra comienza como una imagen completamente negra.
    imagen = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    # La posición, el tamaño y la rotación cambian en cada imagen.
    centro = (
        int(ANCHO / 2 + rng.integers(-55, 56)),
        int(ALTO / 2 + rng.integers(-55, 56)),
    )
    tamano = int(rng.integers(85, 151))
    angulo = float(rng.uniform(0, 360))

    # Las pequeñas variaciones de proporción evitan generar copias idénticas.
    if nombre == "circulo":
        proporcion = float(rng.uniform(0.92, 1.08))
        ejes = (tamano, max(1, int(tamano * proporcion)))
        cv2.ellipse(imagen, centro, ejes, angulo, 0, 360, 255, -1)
    elif nombre == "cuadrado":
        proporcion = float(rng.uniform(0.90, 1.10))
        rectangulo = (centro, (tamano * 2, tamano * 2 * proporcion), angulo)
        vertices = np.int32(cv2.boxPoints(rectangulo))
        cv2.fillPoly(imagen, [vertices], 255)
    elif nombre == "triangulo":
        vertices = []
        for desplazamiento in (0, 120, 240):
            radio = tamano * float(rng.uniform(0.90, 1.10))
            vertices.append(punto_rotado(centro, radio, angulo + desplazamiento))
        cv2.fillPoly(imagen, [np.int32(vertices)], 255)
    else:
        raise ValueError(f"Forma desconocida: {nombre}")

    # Simula pequeñas variaciones producidas por foco, threshold y morfología.
    desenfoque = int(rng.choice([1, 3, 5]))
    if desenfoque > 1:
        imagen = cv2.GaussianBlur(imagen, (desenfoque, desenfoque), 0)
    _, imagen = cv2.threshold(imagen, int(rng.integers(105, 151)), 255, cv2.THRESH_BINARY)
    kernel_size = int(rng.choice([1, 3, 5]))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        operacion = cv2.MORPH_OPEN if rng.random() < 0.5 else cv2.MORPH_CLOSE
        imagen = cv2.morphologyEx(imagen, operacion, kernel)
    return imagen


def contorno_principal(imagen):
    """Devuelve el contorno de mayor área de una imagen sintética."""
    contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise RuntimeError("No se pudo extraer el contorno de una imagen sintética.")
    return max(contornos, key=cv2.contourArea)


def main():
    # Por defecto se generan 30 muestras reproducibles de cada forma.
    parser = argparse.ArgumentParser(description="Generador automático del dataset")
    parser.add_argument("--cantidad", type=int, default=30, help="muestras por clase")
    parser.add_argument("--salida", default="data/dataset.csv")
    parser.add_argument("--imagenes", default="data/imagenes_sinteticas")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    if args.cantidad < 2:
        raise ValueError("Se necesitan al menos dos muestras por clase.")

    # La semilla permite volver a generar exactamente el mismo dataset.
    rng = np.random.default_rng(args.semilla)
    salida = Path(args.salida)
    carpeta_imagenes = Path(args.imagenes)
    salida.parent.mkdir(parents=True, exist_ok=True)
    carpeta_imagenes.mkdir(parents=True, exist_ok=True)

    filas = []
    # Cada carpeta representa una clase y cada imagen produce una fila del CSV.
    for etiqueta, nombre in ETIQUETAS.items():
        carpeta_clase = carpeta_imagenes / nombre
        carpeta_clase.mkdir(parents=True, exist_ok=True)
        for numero in range(1, args.cantidad + 1):
            imagen = dibujar_figura(nombre, rng)
            ruta_imagen = carpeta_clase / f"{nombre}_{numero:03d}.png"
            if not cv2.imwrite(str(ruta_imagen), imagen):
                raise RuntimeError(f"No se pudo guardar {ruta_imagen}")
            # La muestra contiene siete invariantes y la etiqueta correcta.
            hu = invariantes_hu(contorno_principal(imagen))
            filas.append([*hu.tolist(), etiqueta])

    # Se mezclan las clases antes de guardar el dataset.
    rng.shuffle(filas)
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(ENCABEZADO)
        escritor.writerows(filas)

    print(f"Dataset generado en {salida}")
    print(f"Imágenes generadas en {carpeta_imagenes}")
    print(f"Total: {len(filas)} muestras ({args.cantidad} por clase)")


if __name__ == "__main__":
    main()
