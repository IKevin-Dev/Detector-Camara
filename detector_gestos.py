import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from collections import deque  
import numpy as np
import urllib.request
import os

# Descargar el modelo si no existe
modelo_path = "hand_landmarker.task"
if not os.path.exists(modelo_path):
    print("Descargando modelo... (solo la primera vez)")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, modelo_path)
    print("Modelo descargado!")


def contar_dedos(landmarks, mano_tipo):
    dedos = []
    puntas = [4, 8, 12, 16, 20]

    margen = 0.02

    # lógica del pulgar invertida 
    if mano_tipo == "Right":
        dedos.append(1 if landmarks[4].x > landmarks[3].x + margen else 0)
    else:
        dedos.append(1 if landmarks[4].x < landmarks[3].x - margen else 0)

    # Logica demás dedos
    for punta in puntas[1:]:
        dedos.append(1 if landmarks[punta].y < landmarks[punta - 2].y - margen else 0)

    return dedos


def identificar_gesto(total_dedos, dedos):
    if total_dedos == 0:
        return "Puno"
    elif total_dedos == 5:
        return "Mano abierta"
    elif total_dedos == 1 and dedos[1]:
        return "Apuntando"
    elif total_dedos == 2 and dedos[1] and dedos[2]:
        return "Paz / Victoria"
    elif total_dedos == 1 and dedos[0]:
        return "Pulgar arriba"
    else:
        return f"{total_dedos} dedos"


def dibujar_mano(frame, hand_landmarks):
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        frame,
        proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style()
    )


#parámetros de confianza más exigentes
opciones = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=modelo_path),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(opciones)

historiales = [deque(maxlen=5), deque(maxlen=5)]

cap = cv2.VideoCapture(0)
print("Detector de gestos iniciado. Presiona 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect(mp_image)

    if resultado.hand_landmarks:
        for i, hand_landmarks in enumerate(resultado.hand_landmarks):

            # Dibujar la mano
            dibujar_mano(frame, hand_landmarks)

            # Obtener tipo de mano
            mano_tipo = resultado.handedness[i][0].category_name

            # Contar dedos e identificar gesto
            dedos = contar_dedos(hand_landmarks, mano_tipo)
            total = sum(dedos)
            gesto = identificar_gesto(total, dedos)

          
            historiales[i].append(gesto)
            gesto_estable = max(set(historiales[i]), key=historiales[i].count)

            # Posición para mostrar el texto
            x = int(hand_landmarks[0].x * frame.shape[1])
            y = int(hand_landmarks[0].y * frame.shape[0])

            # Mostrar gesto 
            cv2.putText(frame, gesto_estable, (x - 30, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Deteccion de mano
            label_mano = "Izq" if mano_tipo == "Right" else "Der"
            cv2.putText(frame, label_mano, (x - 30, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    else:
        #  limpiar historiales si no hay manos en pantalla
        for h in historiales:
            h.clear()

        cv2.putText(frame, "Sin manos detectadas", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    cv2.imshow("Detector de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()