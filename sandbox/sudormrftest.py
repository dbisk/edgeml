"""
Experiments with the sudormrf model.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
# import torch.profiler as profiler

from models import sudormrf

# prepare fake data 
dummy_input = torch.rand(1, 1, 8000)
dummy_targets = torch.rand(1, 2, 8000)

# initialize the sudormrf model
model = sudormrf.SuDORMRF(
    out_channels=256,
    in_channels=512,
    num_blocks=8,
    upsampling_depth=4,
    enc_kernel_size=21,
    enc_num_basis=512,
    num_sources=2
)

# send the model to the GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")
model = model.to(device)

# send the data to the GPU if possible
dummy_input = dummy_input.to(device)
dummy_targets = dummy_targets.to(device)

# define a totally fake loss function & optimizer
criterion = lambda x, y: torch.mean(torch.abs(x - y))
optimizer = torch.optim.Adam(model.parameters())

# profile a backwards pass
# TODO: figure out why this doesn't work
# with profiler.profile(activities=[profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     with profiler.record_function("backwards_pass"):
#         optimizer.zero_grad()
#         est_sources = model(dummy_input)
#         l = criterion(est_sources, dummy_targets)
#         l.backward()
#         optimizer.step()
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# use memory_profiler to profile memory instead
try:
    @profile
    def forward_pass(model, dummy_input):
        with torch.no_grad():
            estimated_sources = model(dummy_input)
        return estimated_sources


    @profile
    def backward_pass(model, dummy_input, dummy_targets):
        for i in range(5):
            optimizer.zero_grad()
            estimated_sources = model(dummy_input)
            loss = criterion(estimated_sources, dummy_targets)
            loss.backward()
            optimizer.step()

    
    estimated_sources = forward_pass(model, dummy_input)
    backward_pass(model, dummy_input, dummy_targets)
except NameError:
    print(f"Failed to instantiate @profile decorator. Was this script run with `python3 -m memory_profiler sudormrftest.py`?")
