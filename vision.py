"""Funciones comunes de segmentacion y descripcion de contornos."""

import cv2
import numpy as np


def crear_mascara(frame, umbral: int, kernel_size: int, invertir: bool = False):
    """Convierte un fotograma en una máscara binaria limpia."""
    # Trabajar con un solo canal simplifica la separación entre figura y fondo.
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # La inversión permite usar objetos oscuros sobre un fondo claro.
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(gris, umbral, 255, tipo)
    # La apertura morfológica elimina pequeños puntos blancos de ruido.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, kernel_size),) * 2)
    return gris, binaria, cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)


def contornos_validos(mascara, area_minima: int, descartar_borde: bool = False):
    """Obtiene los contornos externos que cumplen los filtros configurados."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Los contornos pequeños suelen corresponder a ruido o reflejos.
    validos = [c for c in contornos if cv2.contourArea(c) >= max(1, area_minima)]
    if descartar_borde:
        alto, ancho = mascara.shape[:2]

        def queda_dentro(contorno):
            x, y, ancho_contorno, alto_contorno = cv2.boundingRect(contorno)
            return (
                x > 1
                and y > 1
                and x + ancho_contorno < ancho - 1
                and y + alto_contorno < alto - 1
            )

        # Evita interpretar el fondo completo como si fuera una figura.
        validos = [c for c in validos if queda_dentro(c)]
    return validos


def invariantes_hu(contorno):
    """Resume la forma de un contorno mediante sus siete invariantes de Hu."""
    return np.nan_to_num(cv2.HuMoments(cv2.moments(contorno)).flatten(), nan=0.0)


def transformar_hu(valores):
    """Amplía la escala de Hu conservando su signo para estabilizar el modelo."""
    valores = np.asarray(valores, dtype=float)
    resultado = np.zeros_like(valores)
    # Se excluyen los ceros para evitar calcular logaritmos indefinidos.
    no_cero = np.abs(valores) > 1e-30
    resultado[no_cero] = -np.sign(valores[no_cero]) * np.log10(
        np.abs(valores[no_cero])
    )
    return resultado
