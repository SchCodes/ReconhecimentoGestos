
import cv2
import mediapipe as mp
import os

def collect_data(output_dir, class_name, num_images=100):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            file_path = os.path.join(output_dir, f"{class_name}_{count}.jpg")
            cv2.imwrite(file_path, frame)
            count += 1
            print(f"Captured {count}/{num_images}")

        cv2.imshow("Hand Gesture Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
collect_data(output_dir="data/raw/positive", class_name="positive", num_images=200)
