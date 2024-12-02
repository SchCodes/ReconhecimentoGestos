
import cv2
import mediapipe as mp
import os
import numpy as np
import time

def collect_data(output_dir, output_dir_landmarks, class_name, num_images=100):

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_landmarks, exist_ok=True)

    landmarks_list = []
    captured_images = []

    count = 0

    # Configuração de quantas fotos por segundo serão capturadas
    photos_per_second = int(input(f'Capturar quantas imagens por segundo? '))

    frame_interval = 1 / photos_per_second  # Intervalo entre as capturas em segundos

    # Variável de controle para o tempo
    last_capture_time = time.time()

    while count < num_images:

        ret, frame = cap.read()

        if not ret:
            print("Erro ao acessar a camera")
            break

        # obtem o tamanho da imagem
        #height, width, channels = frame.shape
        #print(f'Altura: {height}, Largura: {width} e Canais: {channels}')

        # Exibe o frame da câmera
        #cv2.imshow("Captura Gesto da Mao", frame)
        
        # resultados da detecção de mãos
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # se mãos forem detectadas
        if results.multi_hand_landmarks:

            # para cada mão detectada
            for hand_landmarks in results.multi_hand_landmarks:

                # desenhar landmarks e conexões na imagem
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []

                # para cada ponto na/s mão/s detectada/s
                for landmark in hand_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                #landmarks_list.append(landmarks)

        # Exibe o frame da câmera
        cv2.imshow("Captura Gesto da Mao", frame)

        # verifica se chegou o momento de capturar uma nova imagem e landmarks
        current_time = time.time()

        if current_time - last_capture_time >= frame_interval and results.multi_hand_landmarks:

            last_capture_time = current_time

            # salva os landmarks
            landmarks_list.append(landmarks)

            # salva a imagem em um array
            captured_images.append(frame)

            count += 1

            print(f"Capturado: {count}/{num_images}")

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

    # salva os landmarks em um arquivo .npy
    np.save(os.path.join(output_dir_landmarks, f"{class_name}_landmarks.npy"), landmarks_list)

    # Opcional: Salvar as imagens capturadas no disco
    for idx, img in enumerate(captured_images):
        img_name = f"{output_dir}/{class_name}_{idx}.jpg"
        cv2.imwrite(img_name, img)
        print(f"Imagem salva: {img_name}")

# Exemplo de uso
class_name = "positive"
collect_data(output_dir=f"data/raw/categorias/{class_name}", output_dir_landmarks=f"data/raw/landmarks/{class_name}", class_name= class_name, num_images=10)
