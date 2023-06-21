# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi import FastAPI, UploadFile
# from fastapi import File

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import warnings

# app = FastAPI()


# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# @app.get('/')
# def index(request : Request):
#     context={"request" : request}
#     return templates.TemplateResponse("index.html",context)



# import os
# from tensorflow.keras.models import load_model

# def load_emotion_model():
#     model_path = os.path.join("models","emotion_model.h5")
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         return load_model(model_path)


# def detect_emotion(frame, emotion_model,emotions):
#         # Preprocess the frame
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Initialize emotion count dictionary
#         emotion_count = {emotion: 0 for emotion in emotions}

#         # Draw rectangles around the faces and detect emotion
#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y + h, x:x + w]
#             resized_roi = cv2.resize(face_roi, (48, 48))
#             normalized_roi = resized_roi / 255.0
#             reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
#             emotion_prediction = emotion_model.predict(reshaped_roi)
#             emotion_label = emotions[np.argmax(emotion_prediction)]

#             # Increment emotion count
#             emotion_count[emotion_label] += 1

#             # Draw rectangle and emotion label on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         return frame, emotion_count


# @app.post("/upload", response_class=HTMLResponse)
# async def upload_video(video_file: UploadFile = File(...)):
#     emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     fig1 = None
#     fig2 = None

#     # Save the uploaded video file
    
#     with open("videos/video.mp4", "wb") as f:
#         f.write(await video_file.read())

#     # Load the emotion model
#     emotion_model = load_emotion_model()

#     # Read the video file
#     video = cv2.VideoCapture("videos/video.mp4")

#     # Initialize emotion count dictionary
#     total_emotion_count = {emotion: 0 for emotion in emotions}

#     while True:
#         # Read a frame from the video
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Process the frame and detect emotion
#         processed_frame, emotion_count = detect_emotion(frame, emotion_model, emotions)

#         # Increment total emotion count
#         for emotion, count in emotion_count.items():
#             total_emotion_count[emotion] += count

#         # Display the frame with emotions
#         cv2.imshow("Emotion Detection", processed_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()

#     # Display the emotion count
#     for emotion, count in total_emotion_count.items():
#         print(f"{emotion}: {count}")

#     # Get the emotion with the greatest count
#     max_emotion = max(total_emotion_count, key=total_emotion_count.get)
#     max_count = total_emotion_count[max_emotion]

#     # Define the salesman rating barometer labels
#     barometer_labels = ['Worse', 'Bad', 'OK', 'Good', 'Great']

#     # Define the corresponding emotion labels for the barometer
#     barometer_emotions = ['Angry', 'Disgust', 'Fear', 'Neutral', 'Happy']

#     # Determine the rating level based on the emotion
#     rating_level = barometer_labels[barometer_emotions.index(max_emotion)]
#     print(f"Emotion with the greatest count: {max_emotion}")
#     print(f"Count: {max_count}")
#     print(f"Rating Level: {rating_level}")

#     # Filter emotions with more than 30% of the total count
#     filtered_emotions = {emotion: count for emotion, count in total_emotion_count.items() if
#                         count > 0.2 * sum(total_emotion_count.values())}

#     # Create the pie chart
#     labels = filtered_emotions.keys()
#     counts = filtered_emotions.values()

#     fig1, ax1 = plt.subplots()
#     ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
#     ax1.set_title("Emotion Distribution")
#     ax1.axis('equal')

#     # Create a bar chart for the rating level
#     rating_levels = barometer_labels
#     rating_counts = [0] * len(rating_levels)
#     rating_counts[rating_levels.index(rating_level)] = 1

#     fig2, ax2 = plt.subplots()
#     ax2.bar(rating_levels, rating_counts)
#     ax2.set_title("Salesperson Rating")
#     ax2.set_xlabel("Rating Level")
#     ax2.set_ylabel("Count")
#     data = {
#         "filename": video_file.filename,
#         "fig1": fig1,
#         "fig2": fig2,
#         "total_emotion_count": total_emotion_count,
#         "max_emotion": max_emotion,
#         "max_count": max_count,
#         "rating_level": rating_level,
#         "filtered_emotions": filtered_emotions
#     }

#     # Render the HTML template with the data
#     return templates.TemplateResponse("upload.html", {"request": video_file, "data": data})










                # import numpy as np
                # import tensorflow as tf
                # import warnings
                # from fastapi import FastAPI, Request, File, UploadFile
                # from fastapi.responses import HTMLResponse
                # from fastapi.staticfiles import StaticFiles
                # from fastapi.templating import Jinja2Templates
                # import google.cloud.storage


                # app = FastAPI()
                # app.mount("/static", StaticFiles(directory="static"), name="static")
                # templates = Jinja2Templates(directory="templates")

                # # Set your Google Cloud project ID and bucket name
                # project_id = 'cloudkarya-internship'
                # bucket_name = 'csm_project'

                # # Connect to the SQLite database
                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore")
                #     emotion_model = tf.keras.models.load_model('models/emotion_model.h5')

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


                # @app.get('/')
                # def index(request: Request):
                #     context = {"request": request}
                #     return templates.TemplateResponse("index.html", context)


                # @app.post("/upload", response_class=HTMLResponse)
                # async def upload_video(video_file: UploadFile = File(...)):
                #     # Initialize the Google Cloud Storage client
                #     storage_client = storage.Client(project=project_id)

                #     # Get the bucket
                #     bucket = storage_client.get_bucket(bucket_name)

                #     # Set the video file name and local path to save the video
                #     video_file_name = video_file.filename
                #     local_path = '/'

                #     # Specify the path to save the video in the bucket
                #     blob_path = f"videos/{video_file_name}"

                #     # Save the uploaded video file
                #     with open(os.path.join(local_path, video_file_name), "wb") as f:
                #         f.write(await video_file.read())

                #     # Upload the video file to the bucket
                #     blob = bucket.blob(blob_path)
                #     blob.upload_from_filename(os.path.join(local_path, video_file_name))

                #     # Read the video file
                #     video_path = os.path.join(local_path, video_file_name)
                #     video = cv2.VideoCapture(video_path)

                #     # Initialize emotion count dictionary
                #     total_emotion_count = {emotion: 0 for emotion in emotions}

                #     fig1 = None
                #     fig2 = None

                #     while True:
                #         # Read a frame from the video
                #         ret, frame = video.read()
                #         if not ret:
                #             break

                #         # Process the frame and detect emotion
                #         processed_frame, emotion_count = detect_emotion(frame)

                #         # Increment total emotion count
                #         for emotion, count in emotion_count.items():
                #             total_emotion_count[emotion] += count

                #         # Display the frame with emotions
                #         cv2.imshow("Emotion Detection", processed_frame)
                #         cv2.waitKey(1)

                #     # Release the video capture and destroy windows
                #     video.release()
                #     cv2.destroyAllWindows()

                #     # Generate pie chart for emotion distribution
                #     plt.figure(figsize=(8, 8))
                #     plt.title("Emotion Distribution")
                #     plt.pie(list(total_emotion_count.values()), labels=list(total_emotion_count.keys()), autopct='%1.1f%%')
                #     plt.savefig("static/emotion_pie.png")

                #     # Render the result page with the emotion distribution pie chart
                #     context = {
                #         "request": request,
                #         "video_path": video_path,
                #         "emotion_pie_path": "/static/emotion_pie.png",
                #         "emotion_count": total_emotion_count
                #     }
                #     return templates.TemplateResponse("result.html", context)





















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


from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import speech_recognition as sr
from moviepy.editor import VideoFileClip

from pydub import AudioSegment
from spectralcluster import SpectralClusterer
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import os

import nltk
import openai

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from google.cloud import secretmanager
from google.cloud import bigquery,storage


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ishaansharma/topic-detector")
model = AutoModelForSequenceClassification.from_pretrained("ishaansharma/topic-detector")
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


key_Path = "cloudkarya-internship-76c41ffa6790.json"
project_id = "cloudkarya-internship"
bigquery_Client = bigquery.Client.from_service_account_json(key_Path)
storage_Client = storage.Client.from_service_account_json(key_Path)
bucket_Name = "pkcsm-raw"



@app.get('/')
def index(request : Request):
    context={"request" : request,
             "predicted_topic": "No Video Uploaded"}
    return templates.TemplateResponse("index1.html",context)


@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request : Request, video_file: UploadFile = File(...),text_name: str = Form(...),text_rating: str = Form(...),text_sale: str = Form(...)):
    video_path = f"videos/{video_file.filename}"
    folder_Name = "videos"

    bucket = storage_Client.get_bucket(bucket_Name)
    bucket = storage_Client.get_bucket(bucket_Name)
    blob = bucket.blob(f'{video_file.filename}')
    blob.upload_from_filename(video_path)









    with open(video_path, "wb") as f:
        f.write(await video_file.read())

    def extract_audio(video_path, output_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path)

    audio_path = "audio.wav"
    extract_audio(video_path, audio_path)

    # Perform topic detection on the conversation
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    # Transcribe the audio to text
    conversation = recognizer.recognize_google(audio)

    # Tokenize the conversation
    encoded_input = tokenizer(conversation, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # Get the predicted topic
    predicted_topic = torch.argmax(logits, dim=1).item()

    # Map the predicted topic to the corresponding label
    labels = [
        "Arts_&_culture",
        "Business_&_entrepreneurs",
        "Celebrity_&_pop_culture",
        "Diaries_&_Daily_life",
        "Family",
        "Fashion_&_Style",
        "Film_tv_&_Video",
        "Fitness_&_Health",
        "Food_&_Dining",
        "Gaming",
        "Learning_&_Educational",
        "Music",
        "News_&_Social_concern",
        "Other_hobbies",
        "Relationships",
        "Science_&_Technology",
        "Sports",
        "Travel_&_Adventure",
        "youth_&_student_life"
    ]

    predicted_topic_label = labels[predicted_topic]

   
    wav_fpath = Path(audio_path) 
    wav = preprocess_wav(wav_fpath)
    encoder = VoiceEncoder("cpu")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)


    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=100)

    labels = clusterer.predict(cont_embeds)

    def create_labelling(labels,wav_splits):
        from resemblyzer import sampling_rate
        times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
        labelling = []
        start_time = 0

        for i,time in enumerate(times):
            if i>0 and labels[i]!=labels[i-1]:
                temp = [str(labels[i-1]),start_time,time]
                labelling.append(tuple(temp))
                start_time = time
            if i==len(times)-1:
                temp = [str(labels[i]),start_time,time]
                labelling.append(tuple(temp))
        return labelling
    labelling = create_labelling(labels,wav_splits)

    list=[]
    def split_audio(audio_file, labelling):
        audio = AudioSegment.from_file(audio_file)

        for i, (label, start_time, end_time) in enumerate(labelling):
            start_ms = int(start_time * 1000)  # Convert start time to milliseconds
            end_ms = int(end_time * 1000)  # Convert end time to milliseconds

            segment = audio[start_ms:end_ms]  # Extract the segment

            # Save the segment as a separate audio file
            output_file = f"{label}_segment{i}.wav"
            segment.export(output_file, format="wav")
            list.append(output_file)
            # print(f"Segment {i} saved as {output_file}")

    # Example usage
    audio_file = "audio.wav"
    split_audio(audio_file, labelling)


    passage = [] 
    person="Sales-Person : "
    for i in list:
        recognizer = sr.Recognizer()

        audio_file =i

        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)

            try:
                text=person
                text = text+recognizer.recognize_google(audio)
                if person=="Sales-Person : ":
                   person="Customer : "
                else:
                   person="Sales-Person : "
                passage.append(text)
            except sr.UnknownValueError:
                pass
                #print("Speech recognition could not understand audio.")
            except sr.RequestError as e:
                pass
                #print("Could not request results from the speech recognition service; {0}".format(e))
        
    

    
    recognizer = sr.Recognizer()
    audio_file = "audio.wav"

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            print("Recognized Text:")
            print(text)
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from the speech recognition service; {0}".format(e))


    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    x=sia.polarity_scores(text)
    maximum=max(x['neg'],x['neu'],x['pos'])
    if maximum==x['pos']:
        emotion='positive'
    elif maximum==x['neu']:
        emotion='neutral'
    else:
        emotion='negative'
    text=text+'.Here the emotion of the customer and the sales person is '+emotion
    text=text+'.Give us the final summary of the emotion shown by the customer to the sales person and vice versa'
    openai.api_key = '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'
    
    # # Create a client
    # client = secretmanager.SecretManagerServiceClient()

    # # Specify the name of the secret
    # secret_name = "projects/your-project-id/secrets/your-secret-name/versions/latest"

    # # Access the secret
    # response = client.access_secret_version(request={"name": secret_name})
    # api_key = response.payload.data.decode("UTF-8")

    # # Use the API key in your application
    # print(api_key)
    
    def chat_with_gpt3(prompt):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=50,
            temperature=0.6
        )
        return response.choices[0].text.strip()
    print("Welcome to the ChatGPT! Type 'exit' to end the conversation.")
    user_input = text
    response = chat_with_gpt3(user_input)
  
    # query = f"""
    # INSERT INTO `{project_id}.CSM.csm_data`
    # VALUES ('{predicted_topic_label}', '{passage}', '{response}')
    # """
    # job = bigquery_Client.query(query)
    # job.result() 
    
    query = """
    INSERT INTO `{}.CSM.csm_data`
    VALUES (@predicted_topic_label, @passage, @response,@text_name,@text_rating,@text_sale)
    """.format(project_id)

    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = [
        bigquery.ScalarQueryParameter("predicted_topic_label", "STRING", predicted_topic_label),
        bigquery.ScalarQueryParameter("passage", "STRING", passage),
        bigquery.ScalarQueryParameter("response", "STRING", response),
        bigquery.ScalarQueryParameter("text_name", "STRING", text_name),
        bigquery.ScalarQueryParameter("text_rating", "STRING", text_rating),
        bigquery.ScalarQueryParameter("text_sale", "STRING", text_sale)
    ]

    job = bigquery_Client.query(query, job_config=job_config)
    job.result()





    #  # Define the emotions labels
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Function to load the emotion model
    def load_emotion_model():
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        return tf.keras.models.load_model('emotion_model.h5')

    # Function to process frames and detect emotion
    def detect_emotion(frame, emotion_model):
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

    def index():
        # Get the uploaded video file

        # Load the emotion model
        emotion_model = load_emotion_model()

        # Read the video file
        video = cv2.VideoCapture(video_path)

        # Initialize emotion count dictionary
        total_emotion_count = {emotion: 0 for emotion in emotions}

        while True:
            # Read a frame from the video
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame and detect emotion
            processed_frame, emotion_count = detect_emotion(frame, emotion_model)

            # Increment total emotion count
            for emotion, count in emotion_count.items():
                total_emotion_count[emotion] += count

            # Display the frame with emotions
            # cv2.imshow("Emotion Detection", processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

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
        # print(f"Emotion with the greatest count: {max_emotion}")
        # print(f"Count: {max_count}")
        # print(f"Rating Level: {rating_level}")

        # Filter emotions with more than 30% of the total count
        filtered_emotions = {emotion: count for emotion, count in total_emotion_count.items() if count > 0.3 * sum(total_emotion_count.values())}

        # Create the pie chart
        labels = filtered_emotions.keys()
        counts = filtered_emotions.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Emotion Distribution')

        # Create a bar chart for the rating level
        rating_levels = barometer_labels
        rating_counts = [0] * len(rating_levels)
        rating_counts[rating_levels.index(rating_level)] = 1

        fig2, ax2 = plt.subplots()
        ax2.bar(rating_levels, rating_counts)
        ax2.set_xlabel('Rating Level')
        ax2.set_ylabel('Count')
        ax2.set_title('Salesman Rating Barometer')

        return fig1, fig2
    fig1, fig2 = index()
  

    context = {
        "request": request,
        "video_path": video_path,
        "predicted_topic": predicted_topic_label,
        "passage":passage,
        "response":response,
        "fig1":fig1,
        "fig2":fig2
    }


    return templates.TemplateResponse("result.html", context)