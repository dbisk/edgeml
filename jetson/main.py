"""
main.py - Main orchestrator for the Jetson Nano inference and training engine.

Intended to perform the following tasks:
    - listen to audio from a USB microphone
    - break that audio into n-second chunks
    - perform inference on that audio using some model
    - perform training for the same model with that data point

Uses the threading library so that training can be performed concurrently with
the more I/O constrained microphone inputs.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import queue
import threading
import time

import numpy as np
import torch

from models import sudormrf
from jetson import microphone as mic


def inference_engine(model):
    """
    **THIS FUNCTION INTENDED TO BE RUN AS A DAEMON**

    Function that represents the inference side of concurrent inference and
    training. Automatically records microphone input and pipes that through
    the given model. Must be run as a daemon or will get stuck in an infinite
    loop.

    Args:
        model: `Callable`, usually `torch.nn.Module`
            The model used to perform inference. Must be callable with one
            argument.
    """
    # create the queue of recorded sounds to be processed
    sound_q = queue.Queue()

    # create the sound collecting thread
    sound_th = threading.Thread(target = mic.record_audio, args = [8000, True, sound_q])

    # set the sound collecting thread to a daemon
    # this means it will die when this main process dies
    sound_th.daemon = True

    # start the thread
    sound_th.start()

    # from here on the sound thread should be adding microphone recordings to
    # the sound_q. in the main inference thread we can then perform forward 
    # using the given model
    while (True):
        if sound_q.empty():
            # wait a little bit to see if some input is populated
            time.sleep(1)
        else:
            # pop off the front of the queue and perform inference
            d = sound_q.get()
            d = np.expand_dims(d, 0)
            d = torch.Tensor(d).unsqueeze(0)

            est_sources = model(d)
            print(est_sources.size())


def main():
    # define the model to use
    model = sudormrf.SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2
    )

    # run the engines
    inference_engine(model)


if __name__ == '__main__':
    main()
else:
    raise ImportError("This module should not be imported!")
