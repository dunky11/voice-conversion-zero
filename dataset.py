import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, file_paths, speaker_ids, hop_length, sr, sample_frames):
        assert(len(file_paths) == len(speaker_ids))
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.file_paths = file_paths
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        speaker_id  = self.speaker_ids[index]

        audio = np.load(path + ".wav.npy")
        mel = np.load(path + ".mel.npy")

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        return torch.LongTensor(audio), torch.FloatTensor(mel), torch.LongTensor([speaker_id])
