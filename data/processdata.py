import torchaudio.transforms as T
import torch 
import torchaudio
import torch.nn as nn 
import random



Sample_rate = 22050
dur = 10 
NUM_SAMPLES = Sample_rate * dur
N_MELS = 128

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

genre_to_idx = {g: i for i, g in enumerate(GENRES)}

def load_audio(audio_path):
    wf, sr = torchaudio.load(audio_path)
    if wf.shape[0]>1:
        wf = torch.mean(wf,dim=0,keepdim=True)
    if sr != Sample_rate:
        resampler = T.Resample(sr,Sample_rate)
        wf = resampler(wf)
    return wf 
def fix_length(waveform):
    if waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    else:
        pad = NUM_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform
def fix_length_rand(waveform):
    if waveform.shape[1] > NUM_SAMPLES:
        st = random.randint(0,waveform.shape[1]-NUM_SAMPLES)
        waveform = waveform[:, st:st+NUM_SAMPLES]
    else:
        pad = NUM_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform
    

def mix_stems(stem_paths,stage):
    mix = torch.zeros(1,NUM_SAMPLES)
    for path in stem_paths:
        st = load_audio(path)
        if stage == "train":
            st = fix_length_rand(st)
        else:
            st = fix_length(st)
        mix += st
    return mix/len(mix)


