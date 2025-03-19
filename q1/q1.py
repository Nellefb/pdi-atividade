import cv2
import numpy as np


cap = cv2.VideoCapture('pdi-atividade/q1/q1B.mp4')

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

lower_laranja = np.array([5, 100, 100])
upper_laranja = np.array([15, 255, 255])

lower_azul = np.array([100, 100, 100])
upper_azul = np.array([130, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask_laranja = cv2.inRange(hsv, lower_laranja, upper_laranja)
    mask_azul = cv2.inRange(hsv, lower_azul, upper_azul)
    
    contorno_laranja, _ = cv2.findContours(mask_laranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    max_area = 0
    max_contorno = None
    
    for contorno in contorno_laranja + contorno_azul:
        area = cv2.contourArea(contorno)
        if area > 500: 
            shapes.append(contorno)
            
            if area > max_area:
                max_area = area
                max_contorno = contorno
    
    for contorno in shapes:
        cv2.drawContours(frame, [contorno], -1, (255, 0, 0), 2)
        

    if max_contorno is not None:
        x_max, y_max, w_max, h_max = cv2.boundingRect(max_contorno)
        cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)

    colisao = False
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            x1, y1, w1, h1 = cv2.boundingRect(shapes[i])
            x2, y2, w2, h2 = cv2.boundingRect(shapes[j])
            
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                colisao = True
                cv2.putText(frame, "Colisao detectada", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if colisao:
        cv2.putText(frame, "Alerta de colisao", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Feed", frame)

    # Tecla 'ESC' para sair
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
