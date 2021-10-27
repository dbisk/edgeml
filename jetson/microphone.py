"""
microphone.py - Functions related to capturing audio from a microphone for the
NVIDIA Jetson Nano. 

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import queue
import time

import numpy as np
import scipy.io.wavfile as wavf
import sounddevice as sd

def record_audio(sample_rate: int = 8000, fake: bool = False, q: queue.Queue = None):
    """
    Generalized function for recording audio from the microphone. Allows for
    simulated audio recordings using the `fake` boolean flag as an argument. If
    `fake` is `False`, then attempts to find an available USB microphone and
    records one second of audio at the specified sampling rate. 

    Args:
        sample_rate: int
            Sampling rate to record the audio with.
        fake: bool, default `False`
            Set to `True` for simulated audio, and `False` for recorded audio.
        q: queue.Queue, default `None`
            Queue in which to store the recorded audio in. This is passed as
            an argument in order to allow for returning information to a main
            thread when this function is used in a multithreading application.
            If `None`, ignores this argument and simply returns the recorded
            audio.
    Returns:
        wav
            The one-second audio snippet recorded or simulated.
    """
    # NOTE: this function never returns, and thus MUST BE STARTED AS A THREAD
    if (fake):
        # simulate audio collection
        while (True):
            aud = _fake_audio(sample_rate, 1)
            if q is not None:
                q.put(aud)
            time.sleep(1)
    else:
        # collect audio from the microphone
        while (True):
            aud = _real_audio(sample_rate, 1)
            if q is not None:
                q.put(aud)

def _fake_audio(sample_rate: int, length: float) -> np.ndarray:
    """
    Generates a fake, arbitrary length of audio, where `length` is in seconds.

    Currently creates a 440 Hz sine wave. Subject to change.
    """
    r, d = wavf.read('sine440_8k.wav')
    assert r == sample_rate, f"invalid requested `sampled_rate`, must be {r}."
    assert length < len(d) * 1.0 / r, f"invalid length, must be < {len(d) * 1.0 / r}."
    l = r * length
    return d[:l].astype('float')

def _real_audio(sample_rate: int, length: float) -> np.ndarray:
    """
    Records audio from a microphone of length `length`, in seconds.
    """
    arr = sd.rec(int(length * sample_rate), samplerate=sample_rate, channels=1)
    return arr.flatten().astype('float')
