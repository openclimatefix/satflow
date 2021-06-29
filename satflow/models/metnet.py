from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from satflow.models.base import Model, register_model
from satflow.models.layers.ConvGRU import ConvGRU
from axial_attention import AxialAttention


@register_model
class MetNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return MetNet(config=config)


__all__ = [
    "DownSampler",
    "TemporalEncoder",
    "condition_time",
    "ConditionTime",
    "feat2image",
    "MetNet",
]


def DownSampler(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 160, 3, padding=1),
        nn.MaxPool2d((2, 2), stride=2),
        nn.BatchNorm2d(160),
        nn.Conv2d(160, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.MaxPool2d((2, 2), stride=2),
    )


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])


def condition_time(x, i=0, size=(12, 16), seq_len=15):
    "create one hot encoded time image-layers, i in [1, seq_len]"
    assert i < seq_len
    times = (torch.eye(seq_len, dtype=x.dtype, device=x.device)[i]).unsqueeze(-1).unsqueeze(-1)
    ones = torch.ones(1, *size, dtype=x.dtype, device=x.device)
    return times * ones


class ConditionTime(nn.Module):
    "Condition Time on a stack of images, adds `horizon` channels to image"

    def __init__(self, horizon, ch_dim=2):
        super().__init__()
        self.horizon = horizon
        self.ch_dim = ch_dim

    def forward(self, x, fstep=0):
        "x stack of images, fsteps"
        bs, seq_len, ch, h, w = x.shape
        ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, seq_len, 1, 1, 1)
        x = torch.cat([x, ct], dim=self.ch_dim)
        assert x.shape[self.ch_dim] == (ch + self.horizon)  # check if it makes sense
        return x


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)


class TimeDistributed(nn.Module):
    "Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time."

    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            # only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)

    def low_mem_forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)

    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f"TimeDistributed({self.module})"


def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(
        torch.stack([t[i] for t in tuples], dim=stack_dim) for i in list(range(len(tuples[0])))
    )


class MetNet(nn.Module):
    def __init__(
        self,
        image_encoder,
        hidden_dim,
        ks=3,
        n_layers=1,
        n_att_layers=1,
        head=None,
        horizon=3,
        n_feats=0,
        p=0.2,
        debug=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_feats = n_feats
        self.drop = nn.Dropout(p)
        nf = 256  # from the simple image encoder
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(horizon)
        self.temporal_enc = TemporalEncoder(nf, hidden_dim, ks=ks, n_layers=n_layers)
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=8, num_dimensions=2)
                for _ in range(n_att_layers)
            ]
        )

        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

        self.debug = debug

    def encode_timestep(self, x, fstep=1):
        if self.debug:
            print(f"Encode Timestep:(i={fstep})")
        if self.debug:
            print(f" input shape: {x.shape}")

        # Condition Time
        x = self.ct(x, fstep)
        if self.debug:
            print(f" CondTime->x.shape: {x.shape}")

        ##CNN
        x = self.image_encoder(x)
        if self.debug:
            print(f" encoded images shape: {x.shape}")

        # Temporal Encoder
        _, state = self.temporal_enc(self.drop(x))
        if self.debug:
            print(f" temp_enc out shape: {state.shape}")
        return self.temporal_agg(state)

    def forward(self, imgs, feats):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        - feats [bs, n_feats, seq_len]"""
        if self.debug:
            print(f" Input -> (imgs: {imgs.shape}, feats: {feats.shape})")
        # stack feature as images
        if self.n_feats > 0:
            feats = feat2image(feats, target_size=imgs.shape[-2:])
            imgs = torch.cat([imgs, feats], dim=2)
        if self.debug:
            print(f" augmented imgs:   {imgs.shape}")

        # Compute all timesteps, probably can be parallelized
        res = []
        for i in range(self.horizon):
            x_i = self.encode_timestep(imgs, i)
            out = self.head(x_i)
            res.append(out)
        res = torch.stack(res, dim=1).squeeze()
        if self.debug:
            print(f"{res.shape}")
        return res
