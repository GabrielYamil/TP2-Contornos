import cv2
import numpy as np
import os

os.makedirs("referencias", exist_ok=True)

width = 400
height = 400

triangulo = np.ones((height, width), dtype=np.uint8) * 255

punts = np.array([
    [200, 80],
    [80, 300],
    [320, 300]
])

cv2.fillPoly(triangulo, [punts], 0)

cv2.imwrite("referencias/triangulo.png", triangulo)

cuadrado = np.ones((height, width), dtype=np.uint8) * 255

cv2.rectangle(
    cuadrado,
    (80, 80),
    (320, 320),
    0,
    -1
)

cv2.imwrite("referencias/cuadrado.png", cuadrado)


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