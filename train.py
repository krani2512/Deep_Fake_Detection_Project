import cv2
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model

# -------------------- MesoNet Definition --------------------
def Meso4():
    x = Input(shape=(128, 128, 3))
    y = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(8, (5, 5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(16, (5, 5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(16, (5, 5), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(pool_size=(4, 4), padding='same')(y)

    y = Flatten()(y)
    y = Dense(16)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)
    return Model(inputs=x, outputs=y)

# -------------------- Load Pretrained Weights --------------------
def load_mesonet(weights_path='meso4_weights.h5'):
    model = Meso4()
    model.load_weights(weights_path)
    return model

# -------------------- Face Extraction --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    return face

# -------------------- Frame Sampling and Prediction --------------------
def detect_deepfake(video_path, model, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames_analyzed = 0
    predictions = []

    while cap.isOpened() and frames_analyzed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frames_analyzed % 5 != 0:
            frames_analyzed += 1
            continue
        face = extract_face(frame)
        if face is not None:
            face = np.expand_dims(face, axis=0)
            pred = model.predict(face)[0][0]
            predictions.append(pred)
        frames_analyzed += 1

    cap.release()

    if not predictions:
        return {'label': 'No face detected', 'confidence': 0.0}

    avg_conf = np.mean(predictions)
    label = "Deepfake" if avg_conf > 0.5 else "Real"
    return {'label': label, 'confidence': round(avg_conf * 100, 2)}
