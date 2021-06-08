import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from preprocess import wav_to_log_mel
import librosa

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SpeechDataset
from model import Encoder, DecoderWithTransformer as Decoder
import os
import pandas as pd
import numpy as np
import soundfile as sf

def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))

def cleanup(
    file_paths, 
    speaker_ids, 
    min_samples = 5121
):
    new_file_paths = []
    new_speaker_ids = []
    for path, speaker_id in tqdm(zip(file_paths, speaker_ids)):
        audio = np.load(path + ".wav.npy")
        if audio.shape[0] >= min_samples:
            new_file_paths.append(path)
            new_speaker_ids.append(speaker_id)
    return new_file_paths, new_speaker_ids 

@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=cfg.training.optimizer.lr)
    [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    df = pd.read_csv(
        "/home/tim/Desktop/github/ZeroSpeech/datasets/wav2spk.txt",
        header=None, 
        delimiter=" "
    )
    file_paths = df[0].values
    speaker_ids = df[1].values

    speaker_ids = [el - 1 for el in speaker_ids]

    file_paths = [
        os.path.join(
            "/home/tim/Desktop/github/ZeroSpeech/datasets/flickr_8k", os.path.splitext(el)[0]
        )  for el in file_paths
    ]
    del df
    file_paths, speaker_ids = cleanup(file_paths, speaker_ids)

    dataset = SpeechDataset(
        file_paths,
        speaker_ids,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True
    )

    n_epochs = cfg.training.n_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            encoder.train()
            decoder.train()
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)
            speakers = speakers.squeeze(1)
            optimizer.zero_grad()

            z, vq_loss, perplexity = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)
            
            if global_step % 25 == 0:
                print("epoch:{}, recon loss:{:.2f}, vq loss:{:.2f}, perpexlity:{:.3f}"
                    .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))

            if global_step % 500 == 0:
                encoder.eval()
                decoder.eval()
                audio, _ = librosa.load("/home/tim/Desktop/github/ZeroSpeech/trump_small.wav", sr=16000)
                audio = audio / np.abs(audio).max() * 0.999
                mel = wav_to_log_mel(audio)
                mel = torch.FloatTensor(mel).to(device).unsqueeze(0)
                mel = mel[:,:,:300]
                with torch.no_grad():
                    z, _, _ = encoder(mel)
                    y_pred = decoder.generate(z, speakers[0:1])
                sf.write(
                    f"/home/tim/Desktop/github/ZeroSpeech/samples/{global_step}.flac", 
                    y_pred, 
                    16000
                )


if __name__ == "__main__":
    train_model()
