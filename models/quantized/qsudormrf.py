"""
models.quantized.qsudormrf - Quantized version of the SuDORM-RF model in the
parent directory.

Currently a prototype implementation, this model attempts to take advantage of
either Quantization Aware Training (QAT) or Static Quantizaton using PyTorch's
Eager Mode Quantization. More info found
[in the PyTorch quantization docs](https://pytorch.org/docs/stable/quantization.html).

Note: an issue was found when using this model where some weights would converge
to NaNs and the model would fail to learn anything.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import torch

from ..sudormrf import SuDORMRF

class QSuDORMRF(SuDORMRF):
    """
    Quantized version of SuDORMRF. Current prototype is compatible with
    PyTorch's Eager Mode Quantization, in either the Static Quantization or QAT
    forms. 
    """

    def __init__(self,
                 out_channels = 128,
                 in_channels = 512,
                 num_blocks = 16,
                 upsampling_depth = 4,
                 enc_kernel_size = 21,
                 enc_num_basis = 512,
                 num_sources=2):
        
        # initialize the non-quantized SuDORMRF model
        super(QSuDORMRF, self).__init__(
            out_channels, 
            in_channels, 
            num_blocks, 
            upsampling_depth, 
            enc_kernel_size, 
            enc_num_basis, 
            num_sources
        )

        # add the quantization information
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # change encoder so that we can quantize
        self.encoder_conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_basis,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2
        )
        self.encoder_relu = torch.nn.ReLU()
        self.update_encoder()
    
    def update_encoder(self):
        self.encoder = torch.nn.Sequential(*[self.encoder_conv, self.encoder_relu])
    
    def forward(self, input_wav):
        quantized_wav = self.quant(input_wav)
        quantized_wav = super().forward(quantized_wav)
        return self.dequant(quantized_wav)
