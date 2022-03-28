"""
streamed_model.py - lighter models suited for streaming.

TODO: style changes to follow PEP 8
"""

import torch
import torch.nn as nn


class STFT(nn.Module):
    """Forward STFT layer"""

    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=None,
        return_complex=True
    ):
        super().__init__()
        self.w = nn.Parameter(window, requires_grad=False)
        self.p = {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
            'window': self.w,
            'center': center, 
            'pad_mode': pad_mode,
            'normalized': normalized,
            'onesided': onesided,
            'return_complex': return_complex,
        }

    def forward(self, x):
        f = torch.stft(x[:, 0, :], **self.p)
        return torch.cat([f[:, :, :].real, f[:, 1:-1, :].imag], 1)


class ISTFT(nn.Module):
    """Inverse STFT layer"""
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None, center=True, 
        normalized=False, onesided=None, length=None, return_complex=False):
        super().__init__()
        self.w = nn.Parameter(window, requires_grad=False)
        self.p = {'n_fft':n_fft, 'hop_length':hop_length, 'win_length':win_length, 'window':self.w, 'center':center, 
            'length':length, 'normalized':normalized, 'onesided':onesided, 'return_complex':return_complex}

    def forward(self, x):
        f = x[:, :x.shape[1] // 2 + 1, :] + 1j*0
        f[:, 1:-1, :] += 1j * x[:, x.shape[1]//2+1:, :]
        return torch.istft(f, **self.p).unsqueeze(1)


class MaskedConv1d(nn.Module):
    """
    Convolution with a mask, can used for causal filtering (albeit with
    performance loss)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, groups=1, bias=True, mask=None):
        super().__init__()
        from math import sqrt

        self.p = dict(stride=stride, padding=padding, groups=groups, dilation=dilation)
        self.c = nn.Parameter(torch.randn(out_channels, in_channels//groups, kernel_size)/sqrt(in_channels*kernel_size) )
        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = nn.Parameter(torch.zeros(out_channels), requires_grad=False)

        if mask == None:
            self.m = 1
        elif mask == 'causal':
            self.m =  nn.Parameter((torch.arange(kernel_size) <= kernel_size//2).float()[None,None,:], requires_grad=False)
        else:
            self.m =  nn.Parameter(torch.FloatTensor(mask)[None,None,:], requires_grad=False)

        # Special init
        if in_channels == out_channels:
            if kernel_size == 1:
                self.c.data[:,:,0] = torch.eye(in_channels)
            elif groups == in_channels:
                self.c.data[:, :, :] = 0
                self.c.data[:, :, kernel_size // 2] = 1

    def forward(self, x):
        return nn.functional.conv1d(x, self.c * self.m, bias=self.b, **self.p)


class MLPStream(nn.Module):
    """Simpler ResMLP module"""
    def __init__(self, channels, kernel_size, causal = False):
        super().__init__()

        # Affine transforms
        self.w1 = nn.Parameter(torch.ones(1, channels, 1))
        self.b1 = nn.Parameter(torch.zeros(1, channels, 1))
        self.w2 = nn.Parameter(torch.ones(1, channels, 1))
        self.b2 = nn.Parameter(torch.zeros(1, channels, 1))
        self.w3 = nn.Parameter(torch.ones(1, channels, 1))
        self.b3 = nn.Parameter(torch.zeros(1, channels, 1))
        self.w4 = nn.Parameter(torch.ones(1, channels, 1))
        self.b4 = nn.Parameter(torch.zeros(1, channels, 1))

        # Linear layers
        self.L1 = MaskedConv1d(channels, channels, kernel_size, bias=False, groups=channels, 
            padding=kernel_size//2, mask='causal' if causal == 'causal' else None)
        self.L2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.L3 = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x):
        # Cross-patch sublayer
        x0 = x.clone()
        x = x * self.w1 + self.b1
        x = self.L1(x)
        x = x * self.w2 + self.b2
        x += x0

        # Cross-Channel sublayer
        x0 = x.clone()
        x = x * self.w3 + self.b3
        x = self.L3(nn.functional.gelu(self.L2(x)))
        x = x * self.w4 + self.b4
        return x + x0


class WindowedConvTranspose1d(nn.Module):
    """
    Transpose convolution with a synthesis window to avoid overlap add artifacts
    """
    def __init__(self, in_channels, out_channels, output_padding, kernel_size, stride, padding, groups, bias, window):
        super().__init__()

        self.p = dict(stride=stride, padding=padding, groups=groups, bias=bias)
        self.c = nn.Parameter(torch.randn(in_channels, out_channels//groups, kernel_size))
        self.w = 1 if window == None else nn.Parameter(window[None,None,:], requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.conv_transpose1d(x, self.c*self.w, **self.p)


class BWModel(nn.Module):
    def __init__(self,
            in_channels=1,
            out_channels=1,
            enc_kernel_size=21,
            enc_channels=128,
            latent_channels=128,
            num_blocks=12,
            kernel_size=5,
            global_skip=False,
            use_stft=False,
            causal=False,
            synth_window=False,
            upconv=False,
            activations=nn.functional.gelu
        ):
        super().__init__()

        # Remember these
        self.act = activations
        self.gskip = global_skip

        # Front end
        if use_stft:
            self.encoder = STFT(n_fft=enc_channels, hop_length=enc_channels//4, 
                win_length=enc_channels, window=torch.hann_window(enc_channels, periodic=True))
        else:
            self.encoder = nn.Conv1d(in_channels=in_channels, out_channels=enc_channels,
                kernel_size=enc_kernel_size, stride=enc_kernel_size//2,
                padding=enc_kernel_size//2, bias=None)

        # Transform to latent dimension space
        if enc_channels != latent_channels:
            self.pre = nn.Conv1d(in_channels=enc_channels, out_channels=latent_channels, kernel_size=1)
        else:
            self.pre = None

        # Separation module
        self.un = nn.Sequential(*[
            MLPStream(channels=latent_channels, kernel_size=kernel_size, causal=causal) for _ in range(num_blocks)
        ])

        # Transform to encoder/decoder space
        if out_channels*enc_channels*in_channels != latent_channels:
            self.post = nn.Conv1d(latent_channels, out_channels*enc_channels*in_channels, 1)
        else:
            self.post = None
    
        # Decoder
        if use_stft:
            self.decoder = ISTFT(n_fft=enc_channels, hop_length=enc_channels//4, 
                win_length=enc_channels, window=torch.hann_window(enc_channels, periodic=True))
        elif upconv:
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=enc_kernel_size//2),
                nn.Conv1d(in_channels=enc_channels*out_channels*in_channels, out_channels=out_channels*in_channels,
                    kernel_size=enc_kernel_size, padding=(enc_kernel_size+1)//2, bias=None)
            )
        else:
            self.decoder = WindowedConvTranspose1d(
                in_channels=enc_channels*out_channels*in_channels,
                out_channels=out_channels*in_channels,
                output_padding= enc_kernel_size//2-1,
                kernel_size=enc_kernel_size,
                stride=enc_kernel_size//2,
                padding=enc_kernel_size//2,
                groups=1, bias=None,
                window=None if synth_window == False else torch.hann_window(enc_kernel_size, periodic=False)
            )

    # Forward pass
    def forward(self, x):
        if self.gskip:
           x0 = x.clone()
    
        # Front end
        x = self.act(self.encoder(x))

        # Separation module
        if self.pre != None:
            x = self.pre(x)
        x = self.un(x)
        if self.post != None:
            x = self.post(x)

        # Back end
        if self.gskip:
            x = self.decoder(x)
            return x + x0[..., :x.shape[-1]]
        else:
            return self.decoder(x)
