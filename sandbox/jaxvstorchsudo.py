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
import optax
import torch
from tqdm import tqdm

from models.sudojax import sudormrf as sudojax
from models.sudotorch import sudormrf as sudotorch

# meta testing parameters
NUM_LOOPS = 5
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters for data shapes
BATCH_SIZE = 1
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

    # time the forward pass loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        estimated_sources = net.apply(params, dummy_input)
    et = time.perf_counter()
    avg_elapsed = (et - st)*1.0/NUM_LOOPS
    return avg_elapsed

def test_sudojax_full(data: np.ndarray, gts: np.ndarray):
    # convert the input data to jax array
    dummy_input = jnp.array(data, dtype=jnp.float32)
    dummy_gts = jnp.array(gts, dtype=jnp.float32)

    # create the model and initialize
    net = hk.without_apply_rng(hk.transform(
        lambda x: sudojax.SuDORMRF(
            out_channels=128, in_channels=512, num_blocks=16,
            upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512,
            num_sources=2
        )(x)
    ))
    params = net.init(jax.random.PRNGKey(SEED), dummy_input)
    opt = optax.sgd(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, x, y):
        out = net.apply(params, x)
        return jnp.mean(jnp.abs(out - y))

    @jax.jit
    def update(params, opt_state, batch, labels):
        grads = jax.grad(loss_fn)(params, batch, labels)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # time the forward + backward + optimize loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        params, opt_state = update(params, opt_state, dummy_input, dummy_gts)
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

    # send data to device
    model = model.to(DEVICE)
    dummy_input = dummy_input.to(DEVICE)

    # time the forward pass loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        estimated_sources = model(dummy_input)
    et = time.perf_counter()
    avg_elapsed = (et - st)*1.0/NUM_LOOPS
    return avg_elapsed

def test_sudotorch_full(data: np.ndarray, gts: np.ndarray):
    # convert the input data to torch tensor
    dummy_input = torch.tensor(data, dtype=torch.float32)
    dummy_gts = torch.tensor(gts, dtype=torch.float32)

    # create the model
    model = sudotorch.SuDORMRF(
        out_channels=128, in_channels=512, num_blocks=16, upsampling_depth=4,
        enc_kernel_size=21, enc_num_basis=512, num_sources=2
    )

    # send data to device
    model = model.to(DEVICE)
    dummy_input = dummy_input.to(DEVICE)
    dummy_gts = dummy_gts.to(DEVICE)

    # define loss and optimizer
    loss_fn = lambda x, y,: torch.mean(torch.abs(x - y))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # time the forward + backward + optimize loop
    st = time.perf_counter()
    for i in tqdm(range(NUM_LOOPS)):
        optimizer.zero_grad()
        estimated_sources = model(dummy_input)
        loss = loss_fn(estimated_sources, dummy_gts)
        loss.backward()
        optimizer.step()
    et = time.perf_counter()
    avg_elapsed = (et - st)*1.0/NUM_LOOPS
    return avg_elapsed

# create the same data for both models
fake_data = np.random.rand(BATCH_SIZE, 1, SAMPLES)
fake_gts = np.random.rand(BATCH_SIZE, 2, SAMPLES)

# test sudotorch
print(
    f"Testing torch speed with NUM_LOOPS={NUM_LOOPS}; "
    f"BATCH_SIZE={BATCH_SIZE}; "
    f"DEVICE={DEVICE}; "
)
torchtime = test_sudotorch_full(fake_data, fake_gts)
print(f"{torchtime:.5f}s/it.")

# empty the cuda cache
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

# test sudojax
print(
    f"Testing JAX speed with NUM_LOOPS={NUM_LOOPS}; "
    f"BATCH_SIZE={BATCH_SIZE}; "
    f"DEVICE={DEVICE}; "
)
jaxtime = test_sudojax_full(fake_data, fake_gts)
print(f"{jaxtime:.5f}s/it.")
