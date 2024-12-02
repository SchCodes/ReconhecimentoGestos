import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

categories = ['positivo', 'indicador', 'shaka', 'punho', 'rock']

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_gestures(model):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            preprocessed_image = preprocess_image(frame)
            predictions = model.predict(preprocessed_image)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_category = categories[predicted_index]

            print("Previsões: ", predictions)

            cv2.putText(frame, f'Prediction: {predicted_category}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Real-Time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model('models/cnn_model.h5')
    detect_gestures(model)
