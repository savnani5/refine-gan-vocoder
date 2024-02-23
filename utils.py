import numpy as np
import matplotlib.pyplot as plt
import torch
import torchyin
import torchaudio

def generate_audio_envelope_template(waveform, window_size=100):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    else:
        waveform = np.array(waveform)
    rectified_waveform = np.abs(waveform)
    envelope = np.zeros_like(rectified_waveform)
    for i in range(len(rectified_waveform)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(rectified_waveform), i + window_size // 2)
        envelope[i] = np.mean(rectified_waveform[start_index:end_index])
    
    return envelope

def generate_sine_wave_template(waveform, sample_rate, frequency_scale=1000):

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    t = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))
    frequency = np.abs(waveform) * frequency_scale
    phase = np.cumsum(frequency / sample_rate)
    sine_wave = np.sin(2 * np.pi * phase)
    return sine_wave


def generate_pitch_template(waveform, sample_rate):
    # Ensure waveform is a 1D tensor
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    pitch = torchyin.estimate(waveform, sample_rate=sample_rate)

    # Compute mel spectrogram using torchaudio  --> Bottleneck 
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=256)
    mel_spec = mel_spec_transform(waveform[None, :])

    speech_template = torch.zeros_like(waveform)
    voiced_indices = torch.where(pitch > 0)[0]

    f0s = pitch[voiced_indices]
    pulse_lengths = torch.round(sample_rate / f0s).to(torch.int)

    for i, idx in enumerate(voiced_indices):
        pulse_length = pulse_lengths[i].item() 
        pulse = torch.zeros(pulse_length)
        pulse[0] = 1 
        end_idx = min(len(speech_template), idx + pulse_length)
        speech_template[idx:end_idx] += pulse[:end_idx-idx]

    unvoiced_indices = torch.where(pitch <= 0)[0]
    speech_template[unvoiced_indices] = torch.rand(len(unvoiced_indices)) * 2 - 1 

    return speech_template, pitch, mel_spec.squeeze()


def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)



if __name__=="__main__":
    waveform, sample_rate = torchaudio.load('LJ001-0001.wav')

    # Generate the sine wave speech template
    sine_wave_speech_template = generate_sine_wave_template(waveform[0], sample_rate)
    audio_envelope = generate_audio_envelope_template(waveform[0])

    # Plot the original waveform and the sine wave speech template
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(waveform[0].numpy()[:5000], label='Original Waveform')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(sine_wave_speech_template[:5000], label='Sine Wave Speech Template')
    # plt.subplot(2, 1, 3)
    # plt.plot(audio_envelope[:5000], label='Audio envelope Speech Template')
    # plt.subplot(2, 1, 4)
    # plt.plot(audio_envelope[:5000], label='Pitch Speech template')
    plt.legend()
    plt.show()