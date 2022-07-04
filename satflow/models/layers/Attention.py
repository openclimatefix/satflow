import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class SeparableAttn(nn.Module):
    def __init__(
        self, in_dim, activation=F.relu, pooling_factor=2, padding_mode="constant", padding_value=0
    ):
        super().__init__()
        self.model = nn.Sequential(
            SeparableAttnCell(in_dim, "T", activation, pooling_factor, padding_mode, padding_value),
            SeparableAttnCell(in_dim, "W", activation, pooling_factor, padding_mode, padding_value),
            SeparableAttnCell(in_dim, "H", activation, pooling_factor, padding_mode, padding_value),
        )

    def forward(self, x):

        return self.model(x)


class SeparableAttnCell(nn.Module):
    def __init__(
        self,
        in_dim,
        attn_id=None,
        activation=F.relu,
        pooling_factor=2,
        padding_mode="constant",
        padding_value=0,
    ):
        super().__init__()
        self.attn_id = attn_id
        self.activation = activation

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # only pooling on the first dimension
        self.pooling = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(pooling_factor, 1, 1))
        self.pooling_factor = pooling_factor

        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.gamma = nn.Parameter(torch.zeros((1,)))

        self.softmax = nn.Softmax(dim=-1)

    def init_conv(self, conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):

        batch_size, C, T, W, H = x.size()

        assert T % 2 == 0 and W % 2 == 0 and H % 2 == 0, "T, W, H is not even"

        # TODO attention space consumption
        # query = self.query_conv(x).view(batch_size, -1, T * W).permute(0, 2, 1)  # B x (TW) x (CH)
        #
        # key = self.key_conv(x)  # B x C x T x H x W
        # key = self.pooling(key).view(batch_size, -1, T * H // self.pooling_factor)  # B x (CW) x (TH // 8)
        #
        # if H < W:
        #     query = F.pad(query, [0, C * (W - H)], self.padding_mode, self.padding_value)
        # else:
        #     key = F.pad(key, [0, 0, 0, C * (H - W)], self.padding_mode, self.padding_value)

        if self.attn_id == "T":
            attn_dim = T
            out = x[:]
        elif self.attn_id == "W":
            attn_dim = W
            out = x.transpose(2, 3)
        else:
            attn_dim = H
            out = x.transpose(2, 4)

        query = self.query_conv(out).view(batch_size, attn_dim, -1)  # B x T x (CWH)
        key = self.key_conv(out)  # B x C x T x H x W
        key = self.pooling(key).view(
            batch_size, -1, attn_dim // self.pooling_factor
        )  # B x (CWH) x (T // pl)

        dist = torch.bmm(query, key)  # B x T x (T // 4)
        attn_score = self.softmax(dist)  # B x T x (T // 4)

        value = self.value_conv(out)
        value = self.pooling(value).view(
            batch_size, -1, attn_dim // self.pooling_factor
        )  # B x (CWH) x (T // pl)

        out = torch.bmm(value, attn_score.transpose(2, 1))  # B x (CWH) x T

        if self.attn_id == "T":
            out = out.view(batch_size, C, W, H, T).permute(0, 1, 4, 2, 3)
        elif self.attn_id == "W":
            out = out.view(batch_size, C, T, H, W).permute(0, 1, 2, 4, 3)
        elif self.attn_id == "H":
            out = out.view(batch_size, C, T, W, H)

        out = self.gamma * out + x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation=F.relu, pooling_factor=2):  # TODO for better compability

        super(SelfAttention, self).__init__()
        self.activation = activation

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.pooling = nn.MaxPool3d(kernel_size=2, stride=pooling_factor)
        self.pooling_factor = pooling_factor**3

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def init_conv(self, conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):

        if len(x.size()) == 4:
            batch_size, C, W, H = x.size()
            T = 1
        else:
            batch_size, C, T, W, H = x.size()

        assert T % 2 == 0 and W % 2 == 0 and H % 2 == 0, "T, W, H is not even"

        N = T * W * H

        query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)  # B x N x C

        key = self.key_conv(x)  # B x C x W x H

        key = self.pooling(key).view(batch_size, -1, N // self.pooling_factor)  # B x C x (N // pl)

        dist = torch.bmm(query, key)  # B x N x (N // pl)
        attn_score = self.softmax(dist)  # B x N x (N // pl)

        value = self.value_conv(x)
        value = self.pooling(value).view(
            batch_size, -1, N // self.pooling_factor
        )  # B x C x (N // pl)

        out = torch.bmm(value, attn_score.permute(0, 2, 1))  # B x C x N

        if len(x.size()) == 4:
            out = out.view(batch_size, C, W, H)
        else:
            out = out.view(batch_size, C, T, W, H)

        out = self.gamma * out + x
        return out


class SelfAttention2d(nn.Module):
    r"""Self Attention Module as proposed in the paper `"Self-Attention Generative Adversarial
    Networks by Han Zhang et. al." <https://arxiv.org/abs/1805.08318>`_
    .. math:: attention = softmax((query(x))^T * key(x))
    .. math:: output = \gamma * value(x) * attention + x
    where
    - :math:`query` : 2D Convolution Operation
    - :math:`key` : 2D Convolution Operation
    - :math:`value` : 2D Convolution Operation
    - :math:`x` : Input
    Args:
        input_dims (int): The input channel dimension in the input ``x``.
        output_dims (int, optional): The output channel dimension. If ``None`` the output
            channel value is computed as ``input_dims // 8``. So if the ``input_dims`` is **less
            than 8** then the layer will give an error.
        return_attn (bool, optional): Set it to ``True`` if you want the attention values to be
            returned.
    """

    def __init__(self, input_dims, output_dims=None, return_attn=False):
        output_dims = input_dims // 8 if output_dims is None else output_dims
        if output_dims == 0:
            raise Exception(
                "The output dims corresponding to the input dims is 0. Increase the input\
                            dims to 8 or more. Else specify output_dims"
            )
        super(SelfAttention2d, self).__init__()
        self.query = nn.Conv2d(input_dims, output_dims, 1)
        self.key = nn.Conv2d(input_dims, output_dims, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attn = return_attn

    def forward(self, x):
        r"""Computes the output of the Self Attention Layer
        Args:
            x (torch.Tensor): A 4D Tensor with the channel dimension same as ``input_dims``.
        Returns:
            A tuple of the ``output`` and the ``attention`` if ``return_attn`` is set to ``True``
            else just the ``output`` tensor.
        """
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        if self.return_attn:
            return out, attn
        return out


if __name__ == "__main__":

    self_attn = SelfAttention(16)  # no less than 8
    print(self_attn)

    n_frames = 4

    x = torch.rand(1, 16, n_frames, 32, 32)
    y = self_attn(x)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='self-attention') as w:
    #     w.add_graph(self_attn, [x,])

    del x, y

    sepa_attn = SeparableAttn(64)
    print(sepa_attn)
    x = torch.rand(1, 64, 3, 128, 256)
    y = sepa_attn(x)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='separable-attention') as w:
    #     w.add_graph(self_attn, [x,])
