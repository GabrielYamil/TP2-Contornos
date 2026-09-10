# Proyectos 1 y 2 — Clasificación de contornos

Este repositorio contiene las dos entregas de Visión Artificial. Comparten la segmentación de la webcam y las mismas tres formas (`circulo`, `cuadrado` y `triangulo`), pero utilizan métodos de clasificación diferentes.

## Requisitos e instalación

Requiere Python 3.12, una webcam y Windows PowerShell. Desde la carpeta del repositorio, crear y activar un entorno virtual:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell impide activar el entorno virtual, habilitarlo solamente para la terminal actual y volver a intentarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Cuando el entorno está activo, la línea de comandos comienza con `(venv)`. No es necesario usar el comando `py` ni abrir previamente la aplicación Cámara de Windows.

## Proyecto 1 — `matchShapes`

El Proyecto 1 carga una imagen de referencia por clase, detecta todos los contornos relevantes y los compara mediante `cv2.matchShapes`. Los objetos reconocidos se muestran en verde y los desconocidos en rojo.

```powershell
python main.py
```

Archivos propios del Proyecto 1:

- `main.py`: aplicación de webcam y clasificación.
- `cargar_referencias.py`: carga los contornos conocidos.
- `crear_referencias.py`: vuelve a generar las imágenes de referencia si fuera necesario.
- `referencias/`: círculo, cuadrado y triángulo de referencia.

Los controles permiten ajustar el umbral, el kernel morfológico, el área mínima y la distancia máxima aceptada.

Para una prueba sencilla, usar figuras claras sobre un fondo oscuro. En la ventana binaria, la figura debe verse blanca y el fondo negro. `Esc` cierra la aplicación.

## Proyecto 2 — Machine learning

El Proyecto 2 reemplaza `matchShapes` por un árbol de decisión entrenado con los siete invariantes de Hu. Consta de las tres aplicaciones independientes solicitadas.

### Opción rápida: dataset sintético

Para crear automáticamente imágenes variadas y un dataset equilibrado, ejecutar:

```powershell
python generar_dataset_sintetico.py
```

El script produce 30 imágenes de cada clase en `data/imagenes_sinteticas/` y guarda sus 90 descriptores en `data/dataset.csv`. Las figuras tienen variaciones de posición, tamaño, rotación, proporción, desenfoque y morfología. Luego se puede continuar directamente con `python entrenar.py`.

Las muestras sintéticas permiten reproducir el entrenamiento rápidamente. Para evaluar el modelo con webcam conviene mantener figuras claras sobre un fondo oscuro, igual que en las imágenes generadas.

El control `Invertir` permite usar también una figura oscura sobre un fondo claro. La máscara siempre debe mostrar la figura en blanco y el fondo en negro; los contornos que tocan el borde de la imagen se descartan para evitar clasificar el fondo completo.

### 1. Generar el dataset

```powershell
python generar_descriptores.py
```

En la ventana de captura:

- `1`: círculo.
- `2`: cuadrado.
- `3`: triángulo.
- `Espacio`: guarda los invariantes de Hu del contorno más grande.
- `Esc`: termina la captura.

Conviene capturar al menos 10 muestras por clase, variando posición, escala, rotación y dibujo. Las muestras se agregan a `data/dataset.csv`.

Las teclas deben presionarse mientras una ventana de OpenCV tiene el foco. Cada muestra guardada también se informa en la terminal.

### 2. Entrenar el modelo

```powershell
python entrenar.py
```

El entrenador valida el dataset, informa una exactitud de validación cuando hay suficientes muestras y guarda el modelo en `modelos/clasificador_hu.joblib`.

El dataset debe contener las tres etiquetas y al menos dos muestras de cada una. Para una evaluación más representativa se recomiendan entre 10 y 20 muestras por clase.

### 3. Clasificar con el modelo

```powershell
python clasificador_ml.py
```

El clasificador carga el archivo `.joblib`, calcula los invariantes de Hu de cada contorno y muestra la etiqueta predicha. Esta aplicación no utiliza `matchShapes` ni las imágenes de referencia.

El orden completo del Proyecto 2 es, por lo tanto:

```powershell
python generar_descriptores.py
python entrenar.py
python clasificador_ml.py
```

## Archivos compartidos

- `vision.py`: threshold, morfología, filtrado de contornos e invariantes de Hu.
- `etiquetas.py`: diccionario numérico de etiquetas del Proyecto 2.
- `requirements.txt`: dependencias de ambas entregas.

## Archivos generados para la entrega

Después de completar el Proyecto 2 se generan dos archivos importantes:

- `data/dataset.csv`: muestras y etiquetas utilizadas para entrenar.
- `modelos/clasificador_hu.joblib`: modelo entrenado utilizado para predecir.

Ambos deben conservarse junto con el código al preparar la entrega.
