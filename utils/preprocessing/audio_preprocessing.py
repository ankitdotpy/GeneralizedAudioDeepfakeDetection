import os
import glob

import librosa
import numpy as np

import tensorflow as tf
import tensorflow_io as tfio

def add_reverb(audio,sr,reverb_duration=4,decay_rate=0.5):
    reverb_samples = int(sr*reverb_duration)
    impulse_response = np.exp(-decay_rate*np.arange(reverb_samples))
    impulse_response /= np.sqrt(np.sum(impulse_response**2))
    reverberant_audio = np.convolve(audio,impulse_response,mode='full')
    return reverberant_audio[:len(audio)]

def add_noise(audio,sr,noise_dir,noise_level=0.40):
    noise_files = glob.glob(noise_dir+'/*.ogg')
    noise_file = np.random.choice(noise_files,size=1)
    noise,_ = librosa.load(noise_file[0],sr=sr)
    if len(noise)>len(audio):
        noise = noise[:len(audio)]
    elif len(audio)>len(noise):
        repeat = len(audio)//len(noise)
        remainder = len(audio)%len(noise)
        noise = np.concatenate([noise]*repeat+[noise[:remainder]])

    return audio + noise_level * noise

def read_audio(audio_paths,config,noise_dir=None):
    spectrograms = []
    for path in audio_paths:
        audio,_ = librosa.load(path.decode('utf-8'),sr=config.sr)
        augmented_audio = add_reverb(audio,config.sr)
        if noise_dir is not None:
            augmented_audio = add_noise(augmented_audio,config.sr,noise_dir)
        augmented_audio = tf.cast(augmented_audio,tf.float32)
        spectrogram = tfio.audio.spectrogram(augmented_audio,nfft=config.n_fft,
                                             window=config.window,stride=config.hop_length)
        
        mel_spec = tfio.audio.melscale(spectrogram,rate=config.sr,mels=config.n_mels,
                                       fmin=config.fmin,fmax=config.fmax)
        
        mel_spec_db = tfio.audio.dbscale(mel_spec,top_db=config.top_db)

        mel_spec_masked = tfio.audio.freq_mask(mel_spec_db,param=config.freq_param)

        spectrograms.append(mel_spec_db)

    return np.array(spectrograms,dtype=object)