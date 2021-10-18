"""memorytester.py

Memory usage experimentation for a given model. Currently uses the SuDORM-RF
model found in `models`. Supports both CPU and GPU memory profiling, based on
the output of `torch.cuda.is_available()`. 

There may be certain issues with combined-memory devices, such as the NVIDIA
Jetson Nano and other SBCs with integrated CPU + graphics.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
from tqdm import tqdm

from models import sudormrf

# script should not be imported
if __name__ != '__main__':
    raise RuntimeError("This is a script and should not be imported!")
    exit(1)

# parameters for memory testing
NUM_LOOPS = 10

# parameters for data lengths
BATCH_SIZE = 1
DATA_LENGTH_SEC = 1
SAMPLING_RATE = 8000

# identify if we have a GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

if (device == 'cuda'):
    print(f"Expected CUDA version: {torch.version.cuda}")
    # clear the GPU cache before sending the model
    torch.cuda.empty_cache()
else:
    import psutil
    base_mem = psutil.virtual_memory()[3] / 1.0e9

# initialize the model. future work here may include automatically adjusting
# these parameters so as to find the best model for the memory requirements.
model = sudormrf.SuDORMRF(
    out_channels=256,
    in_channels=512,
    num_blocks=8,
    upsampling_depth=4,
    enc_kernel_size=21,
    enc_num_basis=512,
    num_sources=2
)

# send the model to GPU if applicable
model = model.to(device)

# define a loss function and optimizer. again, these can be bogus since we
# only care about memory usage in this script.
criterion = lambda x, y: torch.mean(torch.abs(x - y))
optimizer = torch.optim.Adam(model.parameters())

# define forward and backward pass functions to call in testing
def forward_pass(model, x) -> torch.Tensor:
    # we can utilize torch.no_grad() in this function, which should
    # significantly lower the amount of memory used compared to backward pass.
    with torch.no_grad():
        out = model(x)
    return out

def backward_pass(model, x, y):
    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

# estimate the amount of memory used for just the model
if (device == 'cuda'):
    print(f"Model-only peak GPU memory: {torch.cuda.max_memory_allocated() / 1.0e9}")
else:
    pre_loop_base_mem = psutil.virtual_memory()[3] / 1.0e9 - base_mem
    print(f"Model-only CPU memory usage: {pre_loop_base_mem}")

# perform memory testing for forward pass
peak_mems = []
for i in tqdm(range(NUM_LOOPS)):

    # reset the baseline memory allocated
    if (device == 'cuda'):
        torch.cuda.reset_max_memory_allocated()
        
    # prepare fake data. we don't need to use any real data for memory profiling
    # since we don't actually care about the model performance.
    dummy_input = torch.rand(BATCH_SIZE, 1, DATA_LENGTH_SEC * SAMPLING_RATE)
    dummy_targets = torch.rand(BATCH_SIZE, 2, DATA_LENGTH_SEC * SAMPLING_RATE)

    dummy_input = dummy_input.to(device)
    dummy_targets = dummy_targets.to(device)

    # perform a forward pass
    forward_pass(model, dummy_input)

    # perform a backward pass
    backward_pass(model, dummy_input, dummy_targets)

    if (device == 'cuda'):
        peak_mems.append(torch.cuda.max_memory_allocated() / 1.0e9)
    else:
        peak_mems.append(psutil.virtual_memory()[3] / 1.0e9 - base_mem)

# print the max of the stats
print(f"Max peak memory usage: {max(peak_mems)}")
