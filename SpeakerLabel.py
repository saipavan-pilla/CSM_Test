import os
from spleeter.separator import Separator
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from pydub import AudioSegment
import speech_recognition as sr

# Load the audio file
audio_path = 'audio.wav'
separator = Separator('spleeter:2stems')
separator.separate_to_file(audio_path, 'output')

#give the file path to your audio file
audio_file_path = '/content/output/audio/vocals.wav'
wav_fpath = Path(audio_file_path)

wav = preprocess_wav(wav_fpath)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
print(cont_embeds.shape)



from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=100,
    p_percentile=0.90,
    gaussian_blur_sigma=1)

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
        print(f"Segment {i} saved as {output_file}")

# Example usage
audio_file = "/content/output/audio/vocals.wav"
split_audio(audio_file, labelling)



person = "Sales-Person : "
for i in list:
    recognizer = sr.Recognizer()

    audio_file = "/content/" + i

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

        try:
            text = person
            text = text + recognizer.recognize_google(audio)
            if person == "Sales-Person : ":
                person = "Customer : "
            else:
                person = "Sales-Person : "
            print(text)
        except sr.UnknownValueError:
            pass
            # print("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            pass
            # print("Could not request results from the speech recognition service; {0}".format(e))


