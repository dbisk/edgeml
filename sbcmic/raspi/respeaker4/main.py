"""
main.py - Main test script for using audio on the Raspberry Pi using the
Respeaker 4-Mic array.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import numpy as np
import scipy.io.wavfile as wavf
import sounddevice as sd


def record_audio(
    sample_rate: int,
    length: float,
    channels: int = 1
) -> np.ndarray:
    """
    Records audio from a microphone of length `length`, in seconds.
    
    Args:
        sample_rate: `int`
            The sample rate to record the audio at.
        length: `float`
            The length of time, in seconds, to record.
    Returns:
        `np.ndarray`
            Numpy array of the recorded data, as `float`.
    """
    arr = sd.rec(
        int(length * sample_rate),
        samplerate=sample_rate,
        channels=channels
    )
    # rec runs in background, but we need to wait until recording is finished
    sd.wait()
    return arr.flatten().astype(np.float32)


def main():
    # record 5 seconds of audio and save to wav file
    sample_rate = 8000
    length = 5.0

    # record
    recording = record_audio(sample_rate, length)

    # save to wav file
    wavf.write('recording.wav', sample_rate, recording)


if __name__ == '__main__':
    main()
