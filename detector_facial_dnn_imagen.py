import cv2
import os
image_path = "resultados/pruebaIndividual_02.jpg"
#-----LEER MODELO DNN-----

#Arquitectura del modelo
prototxt = "model/deploy.prototxt"
#Pesos del modelo
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

#Cargamos el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#-----LECTURA Y PROCESAMIENTO DE IMAGEN A ANALIZAR-----
#image = cv2.imread("multimedia_prueba/pruebaIndividual_01.jpeg")
image = cv2.imread("multimedia_prueba/pruebaIndividual_02.jpg")
#image = cv2.imread("multimedia_prueba/pruebaIndividual_03.jpg")
#image = cv2.imread("multimedia_prueba/pruebaGrupal_01.jpg")
height, width, _ = image.shape
image_resize = cv2.resize(image, (300,300))
print("Original shape: ", image_resize.shape)

#Crear un blob (Preprocesar la imagen)
    ##para no redimensionar la imagen, ponemos 1.0 (la escala),
    ##Mean es una tecnica que combate los cambios de ilumincacion, 
    #los valores seleccionados se obtuvieron del promedio de las intensidades de los pixeles 
    #de conjunto de imagenes del entrenamiento usado para este modelo
blob = cv2.dnn.blobFromImage(image_resize, 1.0, (300,300), (104, 117, 123)) 
print("blob shape: ", blob.shape)

#Se mostrara la imagen preprocesada para ver la diferencia
#Como se crearon 3 canales, los combinaremos para mostrarlos como una sola imagen.
enseniar_blob = cv2.merge([blob[0][0], blob[0][1], blob[0][2]]) 

#-----DETECCIONES Y PREDICCIONES-----
#Estableceremos el blob como la entrada de la red
net.setInput(blob)

#Propagaremos la entrada hacia adelante en la redm para obtener los resultados de las detecciones
detections = net.forward()
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
output_dir = "resultados"
os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
output_path = os.path.join(output_dir, os.path.basename(image_path))
cv2.imwrite(output_path, image)
print(f"Imagen guardada en: {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
