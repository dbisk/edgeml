# Raspberry Pi Setup Steps

## Installing Raspberry Pi OS

For PyTorch, you **must install a 64-bit version of Raspberry Pi OS**. The instructions initially used can be found from [Q-Engineering](https://qengineering.eu/install-raspberry-64-os.html), but are pretty simple and rewritten here.

1. Find the most recent 64-bit Raspberry Pi OS image [here](https://downloads.raspberrypi.org/raspios_arm64/images/), which is direct from raspberrypi.org
2. Use the Raspberry Pi Imager tool to write the zip file to a MicroSD card. Make sure the MicroSD card is at least 32 GB and preferably larger.

When setting up the Raspberry Pi OS, make sure to set SWAP such that RAM + SWAP are at least 8GB. You can do this after installation by running:

```bash
# remove the old dphys version
sudo /etc/init.d/dphys-swapfile stop
sudo apt-get remove --purge dphys-swapfile
# install zram
sudo wget -O /usr/bin/zram.sh https://raw.githubusercontent.com/novaspirit/rpi_zram/master/zram.sh
# set autoload
sudo nano /etc/rc.local
# add the next line before exit 0
/usr/bin/zram.sh &
# save with <Ctrl+X>, <Y> and <Enter>
# then perform these steps
sudo chmod +x /usr/bin/zram.sh
sudo nano /usr/bin/zram.sh
# alter the limit with * 2
mem=$(( ($totalmem / $cores)* 1024 * 2))
# save with <Ctrl+X>, <Y> and <Enter>
sudo reboot
```

## Installing PyTorch

There are two options for installing PyTorch.

1. Follow the instructions for installing via wheel found at [Q-Engineering](https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html).
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

