import torchaudio.transforms as T
import torch 
import torchaudio
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader
import pandas as pd 
from augmentation import NoiseAug
from sklearn.model_selection import train_test_split
from processdata import mix_stems,load_audio,fix_length

Sample_rate = 22050
dur = 10 
import os 
NUM_SAMPLES = Sample_rate * dur
N_MELS = 128

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

genre_to_idx = {g: i for i, g in enumerate(GENRES)}
test_data = pd.read_csv('/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv')


mel_transform = T.MelSpectrogram(
    sample_rate=Sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

amplitude_to_db = T.AmplitudeToDB()

def waveform_to_logmel(waveform):
    mel = mel_transform(waveform)
    log_mel = amplitude_to_db(mel)
    return log_mel


class MyDataset(Dataset):
    def __init__(self, samples , esc_noise_dir=None, augment=False,stage="train"):
        self.augment = augment
        self.samples = samples 
        self.noise_adder = NoiseAug(esc_noise_dir) if esc_noise_dir else None
        self.stage = stage
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem_paths, label = self.samples[idx]

        mixture = mix_stems(stem_paths,self.stage)

        if self.augment and self.noise_adder:
            mixture = self.noise_adder.add_noise(mixture)

        features = waveform_to_logmel(mixture)

        return features, torch.tensor(label)
    

class MyTestData(Dataset):
    def __init__(self, samples):
        self.samples = samples     

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id,audio = self.samples[idx]
        audio = load_audio(audio)
        audio = fix_length(audio)

        features = waveform_to_logmel(audio)

        return features, file_id
    

def Create_dataset(root_dir):
    samples = []
    for genre in GENRES:
            genre_path = os.path.join(root_dir, genre)
            for song in os.listdir(genre_path):
                song_path = os.path.join(genre_path, song)

                stems = [
                    os.path.join(song_path, s)
                    for s in ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]
                ]

                samples.append((stems, genre_to_idx[genre]))
    return samples 

def create_testset(test_data):
    samples = []
    test_dir = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup"
    for fid,file in zip(test_data["id"] ,test_data["filename"]):
        file_path = os.path.join(test_dir,file)
        samples.append((fid,file_path))
    return samples

genres_dir = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/genres_stems"


data = Create_dataset(genres_dir)
labels = [label  for _,label in data]

train_samples , val_samples = train_test_split(data,test_size=0.2,stratify=labels,random_state=42)
    
def get_trainset(aug=False):
    noise_dir = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/ESC-50-master/audio"
    train_data = MyDataset(train_samples,noise_dir,augment=aug,stage="train")

    return train_data

def get_valset():
    noise_dir = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/ESC-50-master/audio"
    val_data = MyDataset(val_samples,noise_dir,augment=False,stage="val")
    return val_data

def get_trainload(aug,batch_size=16):
    data = get_trainset(aug)
    load = DataLoader(data,batch_size=batch_size,shuffle=True,pin_memory=True)
    return load 

def get_valload(batch_size=16):
    data = get_valset()
    load = DataLoader(data,batch_size=batch_size,shuffle=False,pin_memory=True)
    return load 



def get_testload():
    test_set = create_testset(test_data)
    testdata = MyTestData(test_set)
    testload = DataLoader(testdata,batch_size=16,shuffle=False,pin_memory=True)
    return testload