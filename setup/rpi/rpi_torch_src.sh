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