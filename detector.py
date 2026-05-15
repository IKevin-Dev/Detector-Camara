import cv2

# Modelo preentrenado que viene con OpenCV
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Abrir cámara 
cap = cv2.VideoCapture(0)

print("Cámara iniciada. Presiona 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la cámara.")
        break

    # Convertir a escala de grises (el detector lo necesita)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,   # qué tanto se reduce la imagen en cada escala
        minNeighbors=5,    # cuántas detecciones vecinas confirman un rostro
        minSize=(30, 30)   # tamaño mínimo del rostro a detectar
    )

    # Dibujar rectángulo en cada rostro detectado
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Rostro", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar contador
    cv2.putText(frame, f"Rostros detectados: {len(rostros)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Detector de Rostros", frame)

    # Salir de la app con letra determinada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()