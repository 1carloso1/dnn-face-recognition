import cv2
import sys
import os

#-----LEER MODELO DNN-----
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

# Ruta de salida
output_dir = "./resultados"
os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
output_path = os.path.join(output_dir, "webcamPrueba.mp4")

# Cargamos el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#-----LECTURA DE LA WEBCAM-----
s = 0  # Cámara predeterminada
if len(sys.argv) > 1:
    s = int(sys.argv[1])  # Permitir seleccionar otra cámara si se pasa como argumento

win_name = "Detector Facial en Vivo"
source = cv2.VideoCapture(s)

# Obtener dimensiones de la cámara
frame_width = int(source.get(3))
frame_height = int(source.get(4))
fps = int(source.get(cv2.CAP_PROP_FPS))
if fps == 0:  # A veces FPS no se detecta bien, establecer un valor por defecto
    fps = 30

# Configurar el VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    has_frame, frame = source.read()
    if not has_frame:
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

    # Escribir el frame en el archivo de video
    out.write(frame)

    # Mostrar el frame en la pantalla
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
        break

# Liberar recursos
source.release()
out.release()
cv2.destroyAllWindows()

print(f"Video guardado en: {output_path}")
