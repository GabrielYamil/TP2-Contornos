"""Entrena y persiste un árbol de decisión a partir del dataset CSV."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn import tree
from sklearn.model_selection import train_test_split

from etiquetas import ETIQUETAS


def cargar_dataset(ruta):
    with ruta.open(newline="", encoding="utf-8") as archivo: filas = list(csv.DictReader(archivo))
    if not filas: raise ValueError("Dataset vacío: ejecute generar_descriptores.py.")
    x = np.array([[float(f[f"hu{i}"]) for i in range(1, 8)] for f in filas])
    y = np.array([int(f["etiqueta"]) for f in filas])
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Entrenador de clasificador de contornos")
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--modelo", default="modelos/clasificador_hu.joblib")
    args = parser.parse_args()
    x, y = cargar_dataset(Path(args.dataset))
    clases = Counter(y)
    if len(clases) < 3 or any(n < 2 for n in clases.values()):
        raise ValueError("Se requieren al menos tres clases y dos muestras por clase.")
    if len(y) >= 12 and min(clases.values()) >= 3:
        x_ent, x_prueba, y_ent, y_prueba = train_test_split(x, y, test_size=.25, random_state=42, stratify=y)
        print(f"Exactitud de validación: {tree.DecisionTreeClassifier(random_state=42).fit(x_ent, y_ent).score(x_prueba, y_prueba):.1%}")
    clasificador = tree.DecisionTreeClassifier(random_state=42).fit(x, y)
    destino = Path(args.modelo); destino.parent.mkdir(parents=True, exist_ok=True)
    dump({"clasificador": clasificador, "etiquetas": ETIQUETAS}, destino)
    print(f"Modelo guardado en {destino}. Muestras: {len(y)}. Por clase: {dict(clases)}")


if __name__ == "__main__":
    main()
