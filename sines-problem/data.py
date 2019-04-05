import torch
import math
import random

def generate_curves(freqs, timesteps, resolution):
    """
    freqs: array[batchlen, num_freqs]
    timesteps: int
    """
    (batchlen, nfreqs) = freqs.size()
    freqs = freqs.view(batchlen, 1, nfreqs).float()
    amplitudes = freqs.new_empty(freqs.size()).normal_() * 0.3 + 1.0
    phases = freqs.new_empty(freqs.size()).normal_() * 0.8
    times = torch.arange(0.0, timesteps, device=freqs.device) / resolution
    times = times.view(1, timesteps, 1)

    curve = torch.sum(amplitudes * torch.cos(2*math.pi*freqs*times + phases), dim=2)

    # some additionnal processing for non-linearity
    return nfreqs * torch.tanh(3.0 * curve / nfreqs)

def generate_batch(batchlen, freqrg=(1,8), nfreqrg=(1,8), timesteps=200, resolution=100, device=None):
    """
    batchlen: int
    freqrs: tuple(int, int), range of frequency values
    nfreqrg: tuple(int, int), range of number of frequencies
    timesteps: length of the sequences
    resolution: number of timesteps per frequency=1.0
    """
    nfreq = random.randint(nfreqrg[0], nfreqrg[1])
    freqs = torch.randint(freqrg[0], freqrg[1], size=(batchlen, nfreq), device=device)
    return (freqs, generate_curves(freqs, timesteps, resolution))
