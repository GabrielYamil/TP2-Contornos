import cv2
import numpy as np
import os

# Crea la carpeta donde el Proyecto 1 busca las imágenes conocidas.
os.makedirs("referencias", exist_ok=True)

width = 400
height = 400

# Cada referencia comienza como una imagen completamente blanca.
triangulo = np.ones((height, width), dtype=np.uint8) * 255

punts = np.array([
    [200, 80],
    [80, 300],
    [320, 300]
])

# La figura se dibuja rellena y en color negro.
cv2.fillPoly(triangulo, [punts], 0)

cv2.imwrite("referencias/triangulo.png", triangulo)

# Genera la referencia del cuadrado.
cuadrado = np.ones((height, width), dtype=np.uint8) * 255

cv2.rectangle(
    cuadrado,
    (80, 80),
    (320, 320),
    0,
    -1
)

cv2.imwrite("referencias/cuadrado.png", cuadrado)


# Genera la referencia del círculo.
circulo = np.ones((height, width), dtype=np.uint8) * 255

cv2.circle(
    circulo,
    (200, 200),
    120,
    0,
    -1
)

cv2.imwrite("referencias/circulo.png", circulo)

print("Referencias creadas correctamente.")
