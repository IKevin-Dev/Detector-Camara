import cv2
from ultralytics import YOLO

# Cargar modelo preentrenado 
modelo = YOLO("yolo11n.pt")  # "n" = nano, el más rápido y liviano

print("Detector de objetos iniciado. Presiona 'Q' para salir.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos en el frame
    resultados = modelo(frame, verbose=False)

    # Dibujar las detecciones sobre el frame
    frame_anotado = resultados[0].plot()

    # Mostrar cuántos objetos detectó
    n_objetos = len(resultados[0].boxes)
    cv2.putText(frame_anotado, f"Objetos: {n_objetos}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Detector de Objetos", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()