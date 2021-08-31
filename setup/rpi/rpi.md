# Raspberry Pi Setup Steps

## Installing Raspberry Pi OS

Install using normal instructions from [the official website](https://www.raspberrypi.org/software/operating-systems/). Make sure to set SWAP such that RAM + SWAP are at least 8GB.

## Installing PyTorch

There are two options for installing PyTorch.

1. Follow the instructions for installing via wheel found [here](https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html).
2. Install from source. To install from source make sure you set SWAP to be 8GB. Then, run `rpi_torch_src.sh` in folder. `rpi_torch_src.sh` follows the following steps:
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install ninja-build git cmake libopenmpi-dev libomp-dev ccache libopenblas-dev libblas-dev libeigen3-dev
pip3 install -U --user wheel mock pillow
pip3 install -U --user setuptools
git clone -b v1.8.1 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
pip3 install --user -r requirements.txt
export BUILD_CAFFE2_OPS=OFF
export USE_FBGEMM=OFF
export USE_FAKELOWP=OFF
export BUILD_TEST=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=ON
export USE_XNNPACK=ON
export USE_QNNPACK=ON
export MAX_JOBS=4
export USE_OPENCV=OFF
export USE_NCCL=OFF
export USE_SYSTEM_NCCL=OFF
PATH=/usr/lib/ccache:$PATH
python3 setup.py clean
python3 setup.py bdist_wheel
cd dist
pip3 install --user torch-1.8.1a0+56b43f4-cp37-cp37m-linux_aarch64.whl
```

