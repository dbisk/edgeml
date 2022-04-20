"""sudormrf.py

@brief SuDO-RM-RF model, implemented in Haiku/JAX. 

@author Efthymios Tzinis and Dean Biskup
@email <etzinis2@illinois.edu>, <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import haiku as hk
import jax
import jax.numpy as jnp

import math


class Upsample(hk.Module):
    """
    Upsampling module. Only upsamples the last dimension.
    """

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_size = x.shape[-1] * self.scale_factor
        new_shape = x.shape[:-1] + (x_size, )
        assert len(new_shape) == len(x.shape)
        out = jax.image.resize(x, new_shape, 'nearest')
        return out


class PReLU(hk.Module):
    negative_slope_init: float = 0.01
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.leaky_relu(x)
        # x_size = x.shape[-1]
        # negative_slope = hk.get_parameter(
        #     "negative_slope",
        #     shape=[x_size],
        #     dtype=x.dtype,
        #     init=self.negative_slope_init
        # )
        
        # return jnp.where(x >= 0, x, negative_slope * x)


class ConvNormAct(hk.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation.
    """

    def __init__(self, nOut, kSize, stride=1, groups=1):
        super().__init__()
        self.conv = hk.Conv1D(
            nOut, kSize, stride=stride, padding='SAME', with_bias=True,
            feature_group_count=groups, data_format='NCW'
        )
        self.norm = hk.GroupNorm(groups=1, eps=1e-8)
        self.act = PReLU()
    
    def __call__(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(hk.Module):
    """
    This class defines the convolution layer with normalization
    """

    def __init__(self, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = hk.Conv1D(
            nOut, kSize, stride=stride, padding='SAME', with_bias=True,
            feature_group_count=groups, data_format='NCW'
        )
        self.norm = hk.GroupNorm(groups=1, eps=1e-8)
    
    def __call__(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(hk.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self):
        super().__init__()
        self.norm = hk.GroupNorm(groups=1, eps=1e-08)
        self.act = PReLU()

    def __call__(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(hk.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = hk.Conv1D(
            output_channels=nOut, kernel_shape=kSize, stride=stride, rate=d,
            padding='SAME', feature_group_count=groups, 
            with_bias=True, data_format='NCW',
        )

    def __call__(self, input):
        return self.conv(input)


class DilatedConvNorm(hk.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = hk.Conv1D(
            output_channels=nOut, kernel_shape=kSize, stride=stride, rate=d,
            padding='SAME', feature_group_count=groups, 
            with_bias=True, data_format='NCW',
        )
        self.norm = hk.GroupNorm(groups=1, eps=1e-08)
    
    def __call__(self, input):
        output = self.conv(input)
        return self.norm(output)


class UBlock(hk.Module):
    """
    This class defines the Upsampling block, which is based on the following
    principle:
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            nOut=in_channels, kSize=1, stride=1, groups=1,
        )
        self.depth = upsampling_depth
        self.spp_dw = (DilatedConvNorm(
            nOut=in_channels, kSize=5, stride=1, groups=in_channels, d=1
        ), )
        for i in range(1, upsampling_depth):
            stride = 1 if i == 0 else 2
            self.spp_dw = self.spp_dw + (DilatedConvNorm(
                nOut=in_channels, kSize=2*stride + 1, stride=stride, 
                groups=in_channels, d=1
            ), )
        
        if upsampling_depth > 1:
            self.upsampler = Upsample(scale_factor=2)
        
        self.conv_1x1_exp = ConvNorm(out_channels, kSize=1, stride=1, groups=1)
        self.final_norm = NormAct()
        self.module_act = NormAct()

    def __call__(self, x):
        # TODO: convert this to better jax loops
        
        # reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # gather them now in reverse order
        for _ in range(self.depth-1):
            tmp = output.pop(-1)
            resampled_out_k = self.upsampler(tmp)
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)


class SuDORMRF(hk.Module):

    def __init__(
        self, out_channels=128, in_channels=512, num_blocks=16,
        upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512, num_sources=2
    ):
        super(SuDORMRF, self).__init__()

        # number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # front end
        self.encoder = hk.Sequential([
            hk.Conv1D(
                output_channels=enc_num_basis, kernel_shape=enc_kernel_size,
                stride=enc_kernel_size // 2, padding='SAME', data_format='NCW',
            ),
            jax.nn.relu,
        ])

        # norm before the rest, and apply one more dense layer
        self.ln = hk.GroupNorm(groups=1, eps=1e-08)
        self.l1 = hk.Conv1D(
            output_channels=out_channels, kernel_shape=1, data_format='NCW'
        )
        
        # separation module
        self.sm = hk.Sequential([
            UBlock(
                out_channels=out_channels,
                in_channels=in_channels, 
                upsampling_depth=upsampling_depth
            ) for _ in range(num_blocks)
        ])

        if out_channels != enc_num_basis:
            self.reshape_before_masks = hk.Conv1D(
                output_channels=enc_num_basis,
                kernel_shape=1,
                data_format='NCW'
            )
        
        # masks layer
        self.m = hk.Conv2D(
            output_channels=num_sources,
            kernel_shape=(enc_num_basis + 1, 1),
            padding='SAME',
            data_format='NCHW'
        )

        # back end
        self.decoder = hk.Conv1DTranspose(
            output_channels=num_sources,
            kernel_shape=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding='SAME',
            # output_shape=((enc_num_basis*num_sources - 1) *
            #              enc_kernel_size//2 - enc_kernel_size//2 +
            #              enc_kernel_size - 1),
            data_format='NCW'
        )
        self.ln_mask_in = hk.GroupNorm(groups=1, eps=1e-08)
    
    def __call__(self, input_wav):
        # front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # split paths
        # s = x.clone()
        s = x.copy()

        # separation module
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.out_channels != self.enc_num_basis:
            x = self.reshape_before_masks(x)
        
        # get masks and apply them
        x = self.m(jnp.expand_dims(x, 1))
        if self.num_sources == 1:
            x = jax.nn.sigmoid(x)
        else:
            x = jax.nn.softmax(x, axis=1)
        x = x * jnp.expand_dims(s, 1)

        # back end
        estimated_waveforms = self.decoder(x.reshape(
            (x.shape[0], -1, x.shape[-1])
        ))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = jnp.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=x.dtype)
            padded_x.at[..., :x.shape[-1]].set(x)
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == '__main__':
    dummy_input = jax.random.uniform(
        jax.random.PRNGKey(42), shape=(3, 1, 32079), dtype=jnp.float32
    )

    # haiku initialization
    net = hk.without_apply_rng(hk.transform(
        lambda x: SuDORMRF(
            out_channels=128, in_channels=512, num_blocks=16,
            upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512,
            num_sources=2
        )(x)
    ))
    params = net.init(jax.random.PRNGKey(42), dummy_input)
    

    estimated_sources = net.apply(params, dummy_input)
    print(estimated_sources.shape)
