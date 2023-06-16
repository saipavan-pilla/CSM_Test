import cv2
import numpy as np
import facial-emotion-recognition
import streamlit as st

# Load the FER model
model = FER()

# Function to process frames and detect emotion
def detect_emotion(frame):
    # Detect emotions in the frame
    emotions = model.detect_emotions(frame)
    emotion_count = {emotion: 0 for emotion in emotions}

    # Draw rectangles and labels for the detected emotions
    for face in emotions:
        x, y, w, h = face['box']
        emotion = face['emotions']
        emotion_label = max(emotion, key=emotion.get)
        emotion_count[emotion_label] += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, emotion_count

# Display the interface
def main():
    st.title("Customer Emotion Analysis")
    st.write("Upload a video and view the detected emotions in real-time.")

    video_file = st.file_uploader("Upload Video", type=['mp4'])

    if video_file is not None:
        # Convert the file uploader object to a file path
        video_path = video_file.name
        video_path = f"./{video_path}"

        # Read the video file
        video = cv2.VideoCapture(video_path)

        # Initialize emotion count dictionary
        total_emotion_count = {emotion: 0 for emotion in model.emotion_map.keys()}

        st.set_option('deprecation.showPyplotGlobalUse', False)
        while True:
            # Read a frame from the video
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame and detect emotion
            processed_frame, emotion_count = detect_emotion(frame)

            # Increment total emotion count
            for emotion, count in emotion_count.items():
                total_emotion_count[emotion] += count

            # Display the frame with emotions
            st.image(processed_frame, channels="BGR")

        video.release()

        # Display the emotion count
        st.subheader("Emotion Count")
        for emotion, count in total_emotion_count.items():
            st.write(f"{emotion}: {count}")

        # Get the emotion with the greatest count
        max_emotion = max(total_emotion_count, key=total_emotion_count.get)
        max_count = total_emotion_count[max_emotion]

        # Display the emotion with the greatest count
        st.subheader("Emotion with the Greatest Count")
        st.text(f"Emotion: {max_emotion}")
        st.text(f"Count: {max_count}")

if __name__ == '__main__':
    main()
