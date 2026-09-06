TP2 - Detección y clasificación de contornos
Requisitos
Python 3.12
Webcam
Instalación

Crear y activar un entorno virtual:

py -3.12 -m venv venv

Activar el entorno virtual en PowerShell:

.\venv\Scripts\Activate.ps1

Instalar las dependencias:

pip install -r requirements.txt
Ejecución

Ejecutar el programa principal:

python main.py
Controles

El programa permite ajustar mediante barras deslizantes:

Umbral de binarización.
Tamaño del kernel morfológico.
Área mínima de los contornos.
Distancia máxima para clasificar una forma.

Presionar ESC para cerrar el programa.