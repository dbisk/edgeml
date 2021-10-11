"""
Experimentation with quantization in PyTorch.

Currently, this prototype does Eager Mode Quantization, with Static Quantization (Post Training
Quantization (PTQ)).

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch

from models.quantized import qsudormrf

# prepare fake data
dummy_input = torch.rand(1, 1, 8000)
dummy_targets = torch.rand(1, 2, 8000)

# initialize the model
model_fp32 = qsudormrf.QSuDORMRF(
    out_channels = 256,
    in_channels = 512,
    num_blocks = 8,
    upsampling_depth = 4,
    enc_kernel_size = 21,
    enc_num_basis = 512,
    num_sources = 2
)

# define a totally fake loss functions & optimizer
# criterion = lambda x, y: torch.mean(torch.abs(x - y))
# optimizer = torch.optim.Adam(model_fp32.parameters())

## === Quantization Stuff Below === ##
# must set model to eval for Post-Training Quantization
model_fp32.eval()

# attach a global qconfig. 'fbgemm' for x86, 'qnnpack' for ARM.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# fuse modules. this needs to be done manually depending on model architecture.
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['encoder_conv', 'encoder_relu']])
model_fp32_fused.update_encoder()

# prepare the model for static quantization
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
fake_dset = torch.rand(4, 1, 8000)
model_fp32_prepared(fake_dset)

# convert the observed model to a quantized model. this does several things: quantizes the
# weights, computes and stores the scale and bias value to be used with each activation tensor,
# and replaces key operators with quantized implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

# try a forward pass on the model using the memory profiler
try:
    @profile
    def forward_pass(model, input_wav):
        with torch.no_grad():
            estimated_sources = model(input_wav)
        return estimated_sources
    
    estimated_sources = forward_pass(model_int8, dummy_input)
except NameError:
    print("Failed to instantiate @profile decorator. Was this script run with"
          "`python3 -m memory_profiler qsudormrftest.py`?")
