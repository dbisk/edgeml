# Jetson Nano 2GB Setup Steps

## Stats on the Jetson Nano 2GB

The Jetson Nano 2GB is a little Single Board Computer from NVIDIA. It has

- 4-core ARM A57
- 2GB LPDDR4 Memory
- 128 Core NVIDIA Maxwell Architecture GPU

Generally, the Ubuntu-derived OS NVIDIA recommends for the device uses around
400-500 MB of RAM at idle. 

## Installing the NVIDIA OS

The method for installing the Jetson OS is detailed on
[NVIDIA's website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit).
The process is relatively simple: Just burn NVIDIA's provided image to a microSD
card, plug in a monitor and keyboard, and follow the on-screen instructions on
first boot.

## Installing PyTorch

NVIDIA has prebuilt PyTorch wheels available on their website, though they lag
slightly behind in version number. These **only include PyTorch**, and not
`torchvision` and `torchaudio`. Information on how to find the `torch` wheels
can be found in the
[NVIDIA forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048).
They can also be found on the [Jetson Zoo](https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29).
Once these wheels are downloaded (either via `wget` or by `scp` from another
device), they can be installed with a simple

```bash
python3 -m pip install _____.whl
```

## Compiling and installing TorchVision and TorchAudio

To compile and install `torchvision` or `torchaudio`, follow the following steps:

1. Install the required libraries for compilation:
```bash
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
python3 -m pip install Pillow
```
2. Follow the instructions from the GitHub repository for `torchvision` or
`torchaudio`. For example, for `torchvision`:
```bash
git clone https://github.com/pytorch/vision.git 
cd vision
python3 setup.py install
```
