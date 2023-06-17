from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
from fastapi import File

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request : Request):
    context={"request" : request}
    return templates.TemplateResponse("index.html",context)



import os
from tensorflow.keras.models import load_model

def load_emotion_model():
    model_path = os.path.join("models","emotion_model.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_model(model_path)


def detect_emotion(frame, emotion_model,emotions):
        # Preprocess the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize emotion count dictionary
        emotion_count = {emotion: 0 for emotion in emotions}

        # Draw rectangles around the faces and detect emotion
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            resized_roi = cv2.resize(face_roi, (48, 48))
            normalized_roi = resized_roi / 255.0
            reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
            emotion_prediction = emotion_model.predict(reshaped_roi)
            emotion_label = emotions[np.argmax(emotion_prediction)]

            # Increment emotion count
            emotion_count[emotion_label] += 1

            # Draw rectangle and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, emotion_count


@app.post("/upload", response_class=HTMLResponse)
async def upload_video(video_file: UploadFile = File(...)):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    fig1 = None
    fig2 = None

    # Save the uploaded video file
    
    with open("videos/video.mp4", "wb") as f:
        f.write(await video_file.read())

    # Load the emotion model
    emotion_model = load_emotion_model()

    # Read the video file
    video = cv2.VideoCapture("videos/video.mp4")

    # Initialize emotion count dictionary
    total_emotion_count = {emotion: 0 for emotion in emotions}

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            break

        # Process the frame and detect emotion
        processed_frame, emotion_count = detect_emotion(frame, emotion_model, emotions)

        # Increment total emotion count
        for emotion, count in emotion_count.items():
            total_emotion_count[emotion] += count

        # Display the frame with emotions
        cv2.imshow("Emotion Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Display the emotion count
    for emotion, count in total_emotion_count.items():
        print(f"{emotion}: {count}")

    # Get the emotion with the greatest count
    max_emotion = max(total_emotion_count, key=total_emotion_count.get)
    max_count = total_emotion_count[max_emotion]

    # Define the salesman rating barometer labels
    barometer_labels = ['Worse', 'Bad', 'OK', 'Good', 'Great']

    # Define the corresponding emotion labels for the barometer
    barometer_emotions = ['Angry', 'Disgust', 'Fear', 'Neutral', 'Happy']

    # Determine the rating level based on the emotion
    rating_level = barometer_labels[barometer_emotions.index(max_emotion)]
    print(f"Emotion with the greatest count: {max_emotion}")
    print(f"Count: {max_count}")
    print(f"Rating Level: {rating_level}")

    # Filter emotions with more than 30% of the total count
    filtered_emotions = {emotion: count for emotion, count in total_emotion_count.items() if
                        count > 0.2 * sum(total_emotion_count.values())}

    # Create the pie chart
    labels = filtered_emotions.keys()
    counts = filtered_emotions.values()

    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Emotion Distribution")
    ax1.axis('equal')

    # Create a bar chart for the rating level
    rating_levels = barometer_labels
    rating_counts = [0] * len(rating_levels)
    rating_counts[rating_levels.index(rating_level)] = 1

    fig2, ax2 = plt.subplots()
    ax2.bar(rating_levels, rating_counts)
    ax2.set_title("Salesperson Rating")
    ax2.set_xlabel("Rating Level")
    ax2.set_ylabel("Count")
    data = {
        "filename": video_file.filename,
        "fig1": fig1,
        "fig2": fig2,
        "total_emotion_count": total_emotion_count,
        "max_emotion": max_emotion,
        "max_count": max_count,
        "rating_level": rating_level,
        "filtered_emotions": filtered_emotions
    }

    # Render the HTML template with the data
    return templates.TemplateResponse("upload.html", {"request": video_file, "data": data})




# import cv2
# import base64
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import warnings
# import streamlit as st
# import time
# import sqlite3

# # Connect to the SQLite database
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     emotion_model = tf.keras.models.load_model('emotion_model1.h5')


# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# # img = get_img_as_base64("back.jpeg")

# # page_bg_img = f"""
# # <style>
# # [data-testid="stAppViewContainer"] > .main {{
# # background-image: url("https://images.unsplash.com/photo-1464618663641-bbdd760ae84a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80");
# # background-size: 180%;
# # background-position: top left;
# # background-repeat: no-repeat;
# # background-attachment: local;
# # }}

# # [data-testid="stSidebar"] > div:first-child {{
# # background-image: url("data:image/png;base64,{img}");
# # background-position: center; 
# # background-repeat: no-repeat;
# # background-attachment: fixed;
# # }}

# # [data-testid="stHeader"] {{
# # background: rgba(0,0,0,0);
# # }}

# # [data-testid="stToolbar"] {{
# # right: 2rem;
# # }}
# # </style>
# # """

# # st.markdown(page_bg_img, unsafe_allow_html=True)

# # Define the emotions labels
# emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# # Function to process frames and detect emotion
# def detect_emotion(frame):
#     # Preprocess the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Initialize emotion count dictionary
#     emotion_count = {emotion: 0 for emotion in emotions}

#     # Draw rectangles around the faces and detect emotion
#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y + h, x:x + w]
#         resized_roi = cv2.resize(face_roi, (48, 48))
#         normalized_roi = resized_roi / 255.0
#         reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
#         emotion_prediction = emotion_model.predict(reshaped_roi)
#         emotion_label = emotions[np.argmax(emotion_prediction)]

#         # Increment emotion count
#         emotion_count[emotion_label] += 1

#         # Draw rectangle and emotion label on the frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     return frame, emotion_count


# # Define the salesman rating barometer labels
# barometer_labels = ['Worse', 'Bad', 'OK', 'Good', 'Great']

# # Define the corresponding emotion labels for the barometer
# barometer_emotions = ['Angry', 'Disgust', 'Fear', 'Neutral', 'Happy']


# # Display the interface
# def main():
#     # st.title("Customer Emotion Analysis")
#     # st.write("Upload a video and view the detected emotions in real-time.")

#     video_file = 

#     if video_file is not None:
#         video_path = video_file.name
#         video_path = f"./{video_path}"

#         # Read the video file
#         video = cv2.VideoCapture(video_path)

#         # Initialize emotion count dictionary
#         total_emotion_count = {emotion: 0 for emotion in emotions}

#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         while True:
#             # Read a frame from the video
#             ret, frame = video.read()
#             if not ret:
#                 break

#             # Process the frame and detect emotion
#             processed_frame, emotion_count = detect_emotion(frame)

#             # Increment total emotion count
#             for emotion, count in emotion_count.items():
#                 total_emotion_count[emotion] += count

#             # Display the frame with emotions
#             # st.image(processed_frame, channels="BGR")

#         video.release()

#         # Display the emotion count
#         # st.subheader("Emotion Count")
#         # for emotion, count in total_emotion_count.items():
#         #     st.write(f"{emotion}: {count}")

#         # Get the emotion with the greatest count
#         max_emotion = max(total_emotion_count, key=total_emotion_count.get)
#         max_count = total_emotion_count[max_emotion]

#         # Display the salesman rating barometer
#         st.subheader("Salesperson Rating")
#         st.text(f"Emotion with the greatest count: {max_emotion}")
#         st.text(f"Count: {max_count}")

#         # Determine the rating level based on the emotion
#         rating_level = barometer_labels[barometer_emotions.index(max_emotion)]
#         st.text(f"Rating Level: {rating_level}")

#         # Filter emotions with more than 30% of the total count
#         filtered_emotions = {emotion: count for emotion, count in total_emotion_count.items() if
#                              count > 0.2 * sum(total_emotion_count.values())}

#         # Create the pie chart
#         labels = filtered_emotions.keys()
#         counts = filtered_emotions.values()

#         fig1, ax1 = plt.subplots()
#         ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
#         ax1.set_title("Emotion Distribution")
#         ax1.axis('equal')

#         # Create a bar chart for the rating level
#         rating_levels = barometer_labels
#         rating_counts = [0] * len(rating_levels)
#         rating_counts[rating_levels.index(rating_level)] = 1

#         fig2, ax2 = plt.subplots()
#         ax2.bar(rating_levels, rating_counts)
#         ax2.set_title("Salesperson Rating")
#         ax2.set_xlabel("Rating Level")
#         ax2.set_ylabel("Count")

#         # Display the charts side by side
#         # col1, col2 = st.columns(2)
#         # with col1:
#         #     st.pyplot(fig1)

#         # with col2:
#         #     st.pyplot(fig2)

#         # st.header("Summary of the conversation")
#         # with open("my_text_file.txt", 'r') as file:
#         #     text = file.read()
#         #     st.markdown(text)


# if __name__ == '__main__':
#     main()

