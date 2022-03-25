# Raspberry Pi Setup Steps

Updated 03/25/2022

There is no longer any real setup necessary for the Raspberry Pis. Based on our
experimentation, you'll definitely at least want 4 GB of effective RAM (this
can be actual RAM or through SWAP. SWAP will of course be significantly slower
and wear out your SD card if heavily used.)

The key points of setup are:

1. Make sure the Raspberry Pi is running 64-bit Raspberry Pi OS. This is now
officially supported and can be found on the official Raspberry Pi website.
2. Increase SWAP such that RAM + SWAP is at least 4 GB, if necessary.

### Installing PyTorch

Installing PyTorch on the Raspberry Pi is easy as of PyTorch 1.10. 

1. Install the required libraries for PyTorch through `apt`, namely
```
sudo apt install python3-pip libopenblas-dev libopenmpi-dev libomp-dev
```
2. Install the important python packages for torch.
```
sudo -H pip3 install setuptools==58.3.0
sudo -H pip3 install Cython
```
3. Finally, install torch directly from PyPI.
```
sudo -H pip3 install torch
```

