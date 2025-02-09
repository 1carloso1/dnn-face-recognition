import cv2
from detector_facial_dnn import Dnn_Facial_Detector

def main():
    #-----RUTAS-----
    carpeta_entrada = "multimedia_prueba"
    carpeta_salida = "resultados"

    #-----LEER MODELO DNN-----
        #Arquitectura del modelo
    prototxt = "model/deploy.prototxt"
        #Pesos del modelo
    model = "model/res10_300x300_ssd_iter_140000.caffemodel"
        #Cargamos el modelo
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    facial_detector = Dnn_Facial_Detector(net)

    print("-----WELCOME TO THE FACIAL DETECTOR SYSTEM-----")
    print("-What kind of file do you wanna try?=-")
    print("-1. Image")
    print("-2. Video")
    print("-3. Webcam")
    print("-0. LEAVE")
    choice = int(input("> "))
    
    while choice != 0:
        if choice == 1:
            facial_detector.detector_facial_imagen(carpeta_entrada, carpeta_salida)
            break
        elif choice == 2:
            facial_detector.detector_facial_video(carpeta_entrada, carpeta_salida)
            break
        elif choice == 3:
            facial_detector.detector_facial_webcam(carpeta_salida)
            break

#-----EJECUCION DEL ARCHIVO-----
if __name__ == "__main__":
    main()

