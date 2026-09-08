"""Funciones comunes de segmentacion y descripcion de contornos."""

import cv2
import numpy as np


def crear_mascara(frame, umbral: int, kernel_size: int):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, kernel_size),) * 2)
    return gris, binaria, cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)


def contornos_validos(mascara, area_minima: int):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contornos if cv2.contourArea(c) >= max(1, area_minima)]


def invariantes_hu(contorno):
    return np.nan_to_num(cv2.HuMoments(cv2.moments(contorno)).flatten(), nan=0.0)
