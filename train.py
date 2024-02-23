import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import argparse
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import optim

from dataset import RefineGanDataset, get_dataset_filelist
from models import UNet, MultiPeriodDiscriminator
from dataset import RefineGanDataset
from loss import spectrogram_loss, envelope_loss, feature_loss, generator_loss, discriminator_loss


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class LightningModel(pl.LightningModule):
    def __init__(self, generator, mpd, sr, nfft, hop_len):
        super(LightningModel, self).__init__()
        self.generator = generator
        self.mpd = mpd
        self.sr = sr
        self.nfft = nfft
        self.hop_len = hop_len
        self.sl = spectrogram_loss
        self.el = envelope_loss
        self.discriminator_loss = discriminator_loss
        self.gen_loss = generator_loss

    def training_step(self, batch, batch_idx):
        x, mel, y = batch
        audio_gen = self.generator(x, mel)
        loss_recon = torch.nn.functional.mse_loss(y, audio_gen, reduction='mean')
        loss_envelope = self.el(y, audio_gen)
        return loss_recon+loss_envelope

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    # def forward(self, x, mel):
    #     return self.generator(x, mel)
    
    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     x, mel, y = batch
    #     y_g_hat = self(x, mel)

    #     # Discriminator update
    #     if optimizer_idx == 0:
    #         # MPD
    #         y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
    #         loss_disc_f = self.discriminator_loss(y_df_hat_r, y_df_hat_g)

    #         loss_disc_all = loss_disc_f # + loss_disc_s
    #         return {"loss": loss_disc_all, "log": {"loss_disc": loss_disc_all}}


    #     # Generator update
    #     elif optimizer_idx == 1:
    #         loss_recon = F.l1_loss(y, y_g_hat)
    #         loss_envelope = self.el(y, y_g_hat)

    #         # Feature Matching Losses
    #         fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)[2:4]
    #         # fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)[2:4]
    #         loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
    #         # loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)

    #         # Generator Losses
    #         loss_gen_f = self.generator_loss(self.mpd(y, y_g_hat)[1])
    #         # loss_gen_s = self.generator_loss(self.msd(y, y_g_hat)[1])

    #         loss_gen_all = loss_gen_f + loss_fm_f + loss_recon + loss_envelope
    #         return {"loss": loss_gen_all, "log": {"loss_gen": loss_gen_all}}



    # def configure_optimizers(self):
    #     optim_g = optim.Adam(self.generator.parameters(), lr=1e-3)
    #     optim_d = optim.Adam(self.mpd.parameters(), lr=1e-4)
    #     return [optim_d, optim_g], []



def train(a, h, device):
    # Dataset
    training_filelist, validation_filelist = get_dataset_filelist(a)


    trainset = RefineGanDataset(training_filelist, h.segment_size, h.n_fft, h.hop_size, h.sampling_rate, device)    
    train_dataloader = DataLoader(trainset, batch_size=h.batch_size, shuffle=False)
    

    # for batch_idx, (speech_template, mel_spec, audio) in enumerate(train_dataloader):
    #     # Your processing here
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Speech Template Shape: {speech_template.shape}")
    #     print(f"  Mel Spectrogram Shape: {mel_spec.shape}")
    #     print(f"  Waveform Shape: {audio.shape}")

    # Model initialize
    generator = UNet().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    # Lightning trainer
    lightningmodel = LightningModel(generator, mpd,  h.sampling_rate, h.n_fft, h.hop_size)
    trainer = pl.Trainer(limit_train_batches=1, max_epochs=1)
    trainer.fit(model=lightningmodel, train_dataloaders=train_dataloader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=5, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
   
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(a, h, device)