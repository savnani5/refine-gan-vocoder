import os
import random
import torch
import torchaudio
import torch.utils.data
from utils import generate_pitch_template
from torch.autograd import Variable

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class RefineGanDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, hop_size, sampling_rate, device=None, split=True, shuffle=True):
        self.audio_files = training_files
        random.seed(1)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, original_sample_rate = torchaudio.load(filename)
    
        # Check if resampling is needed
        if original_sample_rate != self.sampling_rate:
            # Initialize the resampler
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sampling_rate)
            audio = resampler(audio)
    
        
        # Normalize the audio
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val


        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        speech_template, pitch, mel_spec = generate_pitch_template(audio, self.sampling_rate)
        
        return Variable(speech_template.unsqueeze(0), requires_grad=True).to(self.device), mel_spec.to(self.device), audio.to(self.device)

    def __len__(self):
        return len(self.audio_files)

