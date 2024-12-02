
import cv2
import tensorflow as tf
import numpy as np

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def detect_gestures(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128)) / 255.0
        input_data = np.expand_dims(resized, axis=(0, -1))

        predictions = model.predict(input_data)
        gesture = np.argmax(predictions)
        confidence = np.max(predictions)

        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
model = load_model("models/cnn_model.h5")
detect_gestures(model)
