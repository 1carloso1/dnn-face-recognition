import cv2
import os
import sys

class Dnn_Facial_Detector():
    def __init__(self, net):
        self.net = net

    def detector_facial_imagen(self, carpeta_entrada, carpeta_salida):
        # Crear carpeta de salida y subcarpeta "image" si no existen
        carpeta_salida = os.path.join(carpeta_salida, "image")
        #-----LECTURA Y PROCESAMIENTO DE IMAGEN A ANALIZAR-----
            # Obtener la lista de imágenes en la carpeta de entrada
        imagenes = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for imagen_nombre in imagenes:
            imagen_ruta = os.path.join(carpeta_entrada, imagen_nombre)
            image = cv2.imread(imagen_ruta)
            height, width, _ = image.shape
            image_resize = cv2.resize(image, (300,300))
            print("Original shape: ", image_resize.shape)
                #Crear un blob (Preprocesar la imagen)
                    #para no redimensionar la imagen, ponemos 1.0 (la escala),
                    #Mean es una tecnica que combate los cambios de ilumincacion, 
                    #los valores seleccionados se obtuvieron del promedio de las intensidades de los pixeles 
                    #de conjunto de imagenes del entrenamiento usado para este modelo
            blob = cv2.dnn.blobFromImage(image_resize, 1.0, (300,300), (104, 117, 123)) 
            print("blob shape: ", blob.shape)
            #       Como se crearon 3 canales, los combinaremos para mostrarlos como una sola imagen.
            enseniar_blob = cv2.merge([blob[0][0], blob[0][1], blob[0][2]]) 
        #-----DETECCIONES Y PREDICCIONES-----
            #Estableceremos el blob como la entrada de la red
            self.net.setInput(blob)
            #Propagaremos la entrada hacia adelante en la redm para obtener los resultados de las detecciones
            detections = self.net.forward()
            print("Forma de la deteccion:", detections.shape)

            #Ahora para ver como funciona de una mejor manera, veremos cada una de las detecciones
            for detection in detections[0][0]:
                    #El primer valor corresponde a la imagen analizada
                    #El segundo es el valor si el rostro fue detectado, mientras mas se acerque al 1 es la precision
                    #Los ultimos corresponden al cuadro delimitador del rostro
                print("Deteccion:", detection)
                #Solo marcaremos las detecciones mayores a 0.5
                if(detection[2] > 0.5): 
                    #Ahora pondremos la deteccion de una manera visual
                    box = detection[3:7] * [width, height, width, height ] 
                    #Los convertiremos en enteros
                    x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    #Con esto tendremos 4 puntos enteros con los cuales podremos visualizar un rectangulo por cada deteccion
                    cv2.rectangle(image, (x_start, y_start), (x_end,y_end), (0, 255, 0), 2)
                    cv2.putText(image, "Precision: {:.2f}".format(detection[2] * 100), (x_start,y_start - 5), 1, 1.2, (0, 255, 255), 2)
            cv2.imshow("Imagen", image)
            # Guardar la imagen con detección
            os.makedirs(carpeta_salida, exist_ok=True)  # Crear la carpeta si no existe
            output_path = os.path.join(carpeta_salida, imagen_nombre)
            cv2.imwrite(output_path, image)
            print(f"Imagen guardada en: {output_path}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detector_facial_video(self, carpeta_entrada, carpeta_salida):
        carpeta_salida = os.path.join(carpeta_salida, "video")
        videos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(('.mp4', '.MOV'))]
        for video_nombre in videos:
            video_ruta = os.path.join(carpeta_entrada, video_nombre)
            # Captura de video
            cap = cv2.VideoCapture(video_ruta)
            # Obtener propiedades del video original
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # Definir la ruta de salida del video
            os.makedirs(carpeta_salida, exist_ok=True)
            output_path = os.path.join(carpeta_salida, video_nombre)
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
                self.net.setInput(blob)
                detections = self.net.forward()
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
            print(f"Video guardado en: {carpeta_salida}")

    def detector_facial_webcam(self, carpeta_salida):
        carpeta_salida = os.path.join(carpeta_salida, "webcam")
        video_nombre = "webcam_face_detection.mp4"
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
        # Definir la ruta de salida del video
        os.makedirs(carpeta_salida, exist_ok=True)
        output_path = os.path.join(carpeta_salida, video_nombre)
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
            self.net.setInput(blob)
            detections = self.net.forward()
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


        