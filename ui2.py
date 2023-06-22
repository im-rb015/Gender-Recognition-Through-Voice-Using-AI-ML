import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os
import pyaudio
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30


def is_silent(snd_data):
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')
        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)
            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for _ in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for _ in range(int(seconds * RATE))])
    return r


def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
    num_silent = 0
    snd_started = False
    r = array('h')

    while True:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)
        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result


class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Gender Recognition through Voice")
        self.geometry("900x600")
        self.configure(background="#EAF6FF")  # Set background color to light blue

        # Load and resize the background image
        background_image = Image.open("GRTV_IMAGE.png")
        resized_background = background_image.resize((900, 600), Image.LANCZOS)
        self.background = ImageTk.PhotoImage(resized_background)

        # Create a label widget for the background image
        self.background_label = tk.Label(self, image=self.background)
        
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)


        self.heading_label = tk.Label(self, text="Gender Recognition Through Voice", font=("Arial", 26, "bold"), bg="#EAF6FF")
        self.heading_label.pack(pady=150)

        self.record_button = tk.Button(self, text="Record", font=("Arial", 14, "bold"), command=self.record)
        self.record_button.pack(pady=20)

        self.select_file_button = tk.Button(self, text="Select File", font=("Arial", 14, "bold"), command=self.select_file)
        self.select_file_button.pack(pady=20)

        self.result_label = tk.Label(self, text="", bg="#EAF6FF")
        self.result_label.pack(pady=10)


    def record(self):
        self.result_label.config(text="Please talk...")
        self.update()

        file_path = "test.wav"
        record_to_file(file_path)
        self.process_audio(file_path)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(initialdir="./", title="Select File",
                                                    filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
        if self.file_path:
            self.process_audio(self.file_path)

    def process_audio(self, file_path):
        features = extract_feature(file_path, mel=True).reshape(1, -1)

        # Load the saved model (after training)
        # model = pickle.load(open("result/mlp_classifier.model", "rb"))
        from utils import create_model

        # Construct the model
        model = create_model()
        # Load the saved/trained weights
        model.load_weights("results/model.h5")

        # Predict the gender
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "MALE" if male_prob > female_prob else "FEMALE"

        # Update the result label
        self.result_label.config(text="Result: " + gender)
        self.update()


if __name__ == "__main__":
    app = Application()
    app.mainloop()

