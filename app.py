import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Emotion Detector")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

emotion_map = {
    0: 0,  # Angry
    1: 0,  # Angry (ค่าใหม่ที่จะแทนที่ 'Disgust' เป็น 'Angry')
    2: 3,  # Sad (ค่าใหม่ที่จะแทนที่ 'Fear' เป็น 'Sad')
    3: 1,  # Happy
    4: 2,  # Neutral
    5: 3,  # Sad
    6: 1,  # Happy (ค่าใหม่ที่จะแทนที่ 'Surprise' เป็น 'Happy')
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button("Analyze"):
        frame = np.array(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            img_gray = gray[y:y + h, x:x + w]
            img_gray = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([img_gray]) != 0:
                img = img_gray.astype('float') / 255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                prediction = classifier.predict(img)[0]
                mapped_prediction = emotion_map[prediction.argmax()]
                label = emotion_labels[mapped_prediction]

                st.write("Emotion:", label)
            else:
                st.write("No Faces")


