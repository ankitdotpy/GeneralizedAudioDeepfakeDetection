import librosa
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

from config import config

def extend_audio(audio,):
    if len(audio)>config.samples:
        audio = audio[:config.samples]
    elif len(audio)<config.samples:
        repeat = config.samples//len(audio)
        remainder = config.samples%len(audio)
        audio = np.concatenate([audio]*repeat+[audio[:remainder]])

    return audio

def add_reverb(audio,sr,reverb_duration=4,decay_rate=0.5):
    reverb_samples = int(sr*reverb_duration)
    impulse_response = np.exp(-decay_rate*np.arange(reverb_samples))
    impulse_response /= np.sqrt(np.sum(impulse_response**2))
    reverberant_audio = np.convolve(audio,impulse_response,mode='full')
    return reverberant_audio[:len(audio)]

def add_noise(audio,sr,noise_level=0.40):
    noise_file = np.random.choice(config.noise_files,size=1)
    noise,_ = librosa.load(noise_file[0],sr=sr)
    if len(noise)>len(audio):
        noise = noise[:len(audio)]
    elif len(audio)>len(noise):
        repeat = len(audio)//len(noise)
        remainder = len(audio)%len(noise)
        noise = np.concatenate([noise]*repeat+[noise[:remainder]])

    return audio + noise_level * noise

def read_audio(audio_path):
    audio,_ = librosa.load(audio_path,sr=config.sr)
    audio = extend_audio(audio)
    audio = tf.cast(audio,tf.float32)
    spectrogram = tfio.audio.spectrogram(audio,nfft=config.n_fft,
                                             window=config.window,stride=config.hop_length)
    spec_masked = tfio.audio.freq_mask(spectrogram,param=config.freq_param)
        
    masked_audio = tfio.audio.inverse_spectrogram(spec_masked, nfft=config.n_fft, window=config.window, 
                                                    stride=config.hop_length, iterations=30)
        
    augmented_audio = add_reverb(masked_audio,config.sr)
    augmented_audio = add_noise(masked_audio,config.sr)
    
    return augmented_audio