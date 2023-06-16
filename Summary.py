from moviepy.editor import VideoFileClip
import speech_recognition as sr
import nltk
import openai


def extract_audio(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)


video_path = "video5.mp4"
output_path = "audio.wav"
extract_audio(video_path, output_path)

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

x = sia.polarity_scores(text)

max = max(x['neg'], x['neu'], x['pos'])
if max == x['pos']:
    emotion = 'positive'
elif max == x['neu']:
    emotion = 'neutral'
else:
    emotion = 'negative'

text = text + '.Here the emotion of the customer and the sales person is ' + emotion
text = text + '.Give us the final summary of the emotion shown by the customer to the sales person and vice versa'

openai.api_key = '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'


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
print("ChatGPT:", response)