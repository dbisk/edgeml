"""
jaxvstorchsudo.py - comparison script for the relative performance of a
Haiku/JAX implementation of the SuDO-RM-RF model compared to a PyTorch
implementation.

Currently only does forward pass.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).absolute().parent.parent))

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

from models.sudojax import sudormrf as sudojax
from models.sudotorch import sudormrf as sudotorch

# meta testing parameters
NUM_LOOPS = 1000
SEED = 42

# parameters for data shapes
BATCH_SIZE = 4
SAMPLES = 16000

# define testing functions
def test_sudojax_forward(data: np.ndarray):
    # convert the input data to jax array
    dummy_input = jnp.array(data, dtype=jnp.float32)

    # create the model and initialize
    net = hk.without_apply_rng(hk.transform(
        lambda x: sudojax.SuDORMRF(
            out_channels=128, in_channels=512, num_blocks=16,
            upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512,
            num_sources=2
        )(x)
    ))
    params = net.init(jax.random.PRNGKey(SEED), dummy_input)

    # utilize jax.jit to increase speed
    @jax.jit
    def forward(p, x):
        return net.apply(p, x)

    # time the forward pass loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        estimated_sources = forward(params, dummy_input)
    et = time.perf_counter()
    avg_elapsed = (et - st)*1.0/NUM_LOOPS
    return avg_elapsed

def test_sudotorch_forward(data: np.ndarray):
    # convert the input data to torch tensor
    dummy_input = torch.tensor(data, dtype=torch.float32)

    # create the model
    model = sudotorch.SuDORMRF(
        out_channels=128, in_channels=512, num_blocks=16, upsampling_depth=4,
        enc_kernel_size=21, enc_num_basis=512, num_sources=2
    )

    if torch.cuda.is_available():
        print("Using CUDA")
        dummy_input = dummy_input.to('cuda')
        model = model.to('cuda')

    # time the forward pass loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        estimated_sources = model(dummy_input)
    et = time.perf_counter()
    avg_elapsed = (et - st)*1.0/NUM_LOOPS
    return avg_elapsed

# create the same data for both models
fake_data = np.random.rand(BATCH_SIZE, 1, SAMPLES)
fake_gts = np.random.rand(BATCH_SIZE, 2, SAMPLES)

# test sudojax
print(f"Testing JAX speed with NUM_LOOPS={NUM_LOOPS}.")
jaxtime = test_sudojax_forward(fake_data)
print(f"{jaxtime:.5f}s/it.")

# test sudotorch
print(f"Testing torch speed with NUM_LOOPS={NUM_LOOPS}.")
torchtime = test_sudotorch_forward(fake_data)
print(f"{torchtime:.5f}s/it.")
