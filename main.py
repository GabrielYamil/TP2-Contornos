import cv2
from cargar_referencias import contornos_referencia

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Threshold")
cv2.namedWindow("Controles")

cv2.createTrackbar("Umbral", "Threshold", 127, 255, nothing)
cv2.createTrackbar("Kernel", "Threshold", 1, 20, nothing)
cv2.createTrackbar("Area Minima", "Controles", 500, 50000, nothing)
cv2.createTrackbar("Distancia Maxima", "Controles", 10, 100, nothing)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener la imagen de la cámara.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    threshold_value = cv2.getTrackbarPos("Umbral", "Threshold")

    _, binary = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    Kernel_size = cv2.getTrackbarPos(
        "Kernel",
        "Threshold"
    )

    if Kernel_size < 1:
       Kernel_size = 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (Kernel_size, Kernel_size)
    )

    morphed = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, hierarchy = cv2.findContours(
        morphed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = cv2.getTrackbarPos(
        "Area Minima",
        "Controles"
    )

    if min_area < 1:
        min_area = 1

    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            filtered_contours.append(contour)

    distancia_slider = cv2.getTrackbarPos(
        "Distancia Maxima",
        "Controles"
    )

    distancia_maxima = distancia_slider / 100.0

    for contour in filtered_contours:

        mejor_nombre = "desconocido"
        mejor_distancia = float('inf')

        for nombre, referencia in contornos_referencia.items():

            distancia = cv2.matchShapes(
                contour,
                referencia,
                cv2.CONTOURS_MATCH_I1,
                0.0
            )

            if distancia < mejor_distancia:

                mejor_distancia = distancia
                mejor_nombre = nombre

        if mejor_distancia <= distancia_maxima:

            texto = (
                f"{mejor_nombre}"
                f"({mejor_distancia:.3f})"
            )

            color = (0, 255, 0)

        else:

            mejor_nombre = "desconocido"

            texto = (
                f"desconocido"
                f"({mejor_distancia:.3f})"
            )

            color = (0, 0, 255)


        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            texto,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Distancia Maxima: {distancia_maxima:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Imagen original",
            frame
        )

        cv2.imshow(
            "Imagen en escala de grises",
            gray
        )

        cv2.imshow(
            "Threshold",
            binary
        )

        cv2.imshow(
            "Morfologia",
            morphed
        )



    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()