import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from utils import stft, get_padding


class UNet(nn.Module):
    def __init__(self, input_channels=1, mel_channels=128, output_channels=1):
        super().__init__()

        rates = [2, 2, 8, 8]
        encoder_channels = [16, 32, 64, 128]

        self.encoder = nn.ModuleList([
            self._encoder_block(input_channels if i == 0 else encoder_channels[i-1], encoder_channels[i], rate, kernel_size=rate*2)
            for i, rate in enumerate(rates)
        ])

        # Assumptions for parallel resblocks
        decoder_channels = [64, 32, 16]
        self.decoder = nn.ModuleList([
            self._decoder_block(mel_channels + encoder_channels[-1], decoder_channels[0], rates[3], kernel_size=rates[3]*2, use_parallel_resblocks=False),
            self._decoder_block(decoder_channels[0] + encoder_channels[-2], decoder_channels[1], rates[2], kernel_size=rates[2]*2, use_parallel_resblocks=True),
            self._decoder_block(decoder_channels[1] + encoder_channels[-3], decoder_channels[2], rates[1], kernel_size=rates[1]*2, use_parallel_resblocks=True),
            self._decoder_block(decoder_channels[2] + encoder_channels[-4], output_channels, rates[0], kernel_size=rates[0]*2, use_parallel_resblocks=False)
        ])

    def _encoder_block(self, in_channels, out_channels, stride, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
            nn.LeakyReLU(),
            ResBlock(out_channels, kernel_sizes=[7])  # Fixed kernel size of 7 in the encoding side
        )

    def _decoder_block(self, in_channels, out_channels, stride, kernel_size, use_parallel_resblocks=False):
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=stride-1),
            nn.LeakyReLU()
        ]
        if use_parallel_resblocks:
            layers.append(ResBlock(out_channels, kernel_sizes=[3, 7, 11]))  # Parallel ResBlocks with different kernel sizes
        else:
            layers.append(ResBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, waveform, mel_spectrogram):
        encoder_outputs = []

        x = waveform
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)

        x = torch.nn.functional.interpolate(mel_spectrogram, size=x.shape[-1], mode='nearest') # important --> try to interpolate less
        x = torch.cat((x, encoder_outputs[-1]), dim=1)

        for i, block in enumerate(self.decoder):
            x = block(x)
            if i < len(self.decoder) - 1:
                resized_x = self.resize_tensor(x, encoder_outputs[-(i+2)])
                x = torch.cat((resized_x, encoder_outputs[-(i+2)]), dim=1)  # Skip concatenation for the last block

        return x
    
    def resize_tensor(self, input_tensor, target_tensor):
        return torch.nn.functional.interpolate(input_tensor, size=target_tensor.shape[-1], mode='nearest')
    

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[7], dilation_sizes=[1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size=ks, padding=ks//2 * dil, dilation=dil))
            for ks in kernel_sizes for dil in dilation_sizes
        ])

    def forward(self, x):
        outputs = 0
        for conv in self.convs:
            outputs += F.leaky_relu(conv(x), 0.1)
        return x + outputs / len(self.convs)

# Multiperiod descriminator
    
class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.conv_post = norm_f(nn.Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)     # real 
            y_d_g, fmap_g = d(y_hat) # fake
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs




if __name__=="__main__":

    dummy = torch.rand(4, 1, 22050)
    mel = torch.rand(4, 128, 87)
    model = UNet()
    out = model(dummy, mel)
    print(out.shape)