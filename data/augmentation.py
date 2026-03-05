import os 
import random
from processdata import load_audio,fix_length
class NoiseAug:
    def __init__(self, esc_audio_dir):
        self.noise_files = [
            os.path.join(esc_audio_dir, f)
            for f in os.listdir(esc_audio_dir)
            if f.endswith(".wav")
        ]

    def add_noise(self, waveform, noise_level=0.2):
        noise_path = random.choice(self.noise_files)
        noise = load_audio(noise_path)
        noise = fix_length(noise)

        return waveform + noise_level * noise