import cv2
import os

#-----LEER MODELO DNN-----
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

# Cargamos el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Ruta de salida
video_path = "multimedia_prueba/pruebaVideo_02.mp4"

# Ruta del video de salida
output_dir = "./resultados"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "pruebaVideo_02_detectado.mp4")

# Captura de video
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video original
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #-----LECTURA Y PROCESAMIENTO DE FRAME A ANALIZAR-----
    height, width, _ = frame.shape
    frame_resize = cv2.resize(frame, (300, 300))

    blob = cv2.dnn.blobFromImage(frame_resize, 1.0, (300, 300), (104, 117, 123))

    #-----DETECCIONES Y PREDICCIONES-----
    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0][0]:
        if detection[2] > 0.5:
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, "Precision: {:.2f}".format(detection[2] * 100), 
                        (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)

    # Escribir el frame con detecciones en el video de salida
    out.write(frame)

    # Mostrar el frame en pantalla
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video guardado en: {output_path}")
