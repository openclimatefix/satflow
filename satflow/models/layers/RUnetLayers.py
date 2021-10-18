"""Layers for RUNet"""
import torch
import torch.nn as nn
from satflow.models.utils import get_conv_layer
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    """
    Initialize network weights

    Args:
        net: network to be initialized
        init_type: options are "normal", "xavier", "kaiming", "orthogonal"
        gain: scaling factor. Default is 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    """
    Convolutional block
    
    A twice-repeated chain of a convolutional layer, batch normalization, and ReLU
    """
    def __init__(self, ch_in, ch_out, conv_type: str = "standard"):
        """
        Initialize the block

        Args:
            ch_in: number of input channels
            ch_out: number of output channels
            conv_type: one of "standard", "coord", "antialiased", or "3d"
        """
        super(conv_block, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.conv = nn.Sequential(
            conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Convolutional block with upsampling

    A chain of an upsample layer with scale factor 2, a convolutional layer,
    batch normalization, and ReLU
    """
    def __init__(self, ch_in, ch_out, conv_type: str = "standard"):
        """
        Initialize the block

        Args:
            ch_in: number of input channels
            ch_out: number of output channels
            conv_type: one of "standard", "coord", "antialiased", or "3d"
        """
        super(up_conv, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    """
    Recurrent block

    A repeated chain of a convolutional layer, batch normalization, and ReLU,
    where the output of the previous step is added to the input for the next step.
    """
    def __init__(self, ch_out, t=2, conv_type: str = "standard"):
        """
        Initialize the block

        Args:
            ch_out: number of channels for input and output
            t: number of steps. Default is 2.
            conv_type: one of "standard", "coord", "antialiased", or "3d"
        """
        super(Recurrent_block, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    """
    Recursive residual CNN block

    A chain of recurrent blocks with a skip connection of the input to the final output
    """
    def __init__(self, ch_in, ch_out, t=2, conv_type: str = "standard"):
        """
        Initialize the block

        Args:
            ch_out: number of input channels
            ch_out: number of output channels
            t: number of steps in the recurrent blocks. Default is 2.
            conv_type: one of "standard", "coord", "antialiased", or "3d"
        """
        super(RRCNN_block, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t, conv_type=conv_type),
            Recurrent_block(ch_out, t=t, conv_type=conv_type),
        )
        self.Conv_1x1 = conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    """
    Convolutional block
    
    A chain of a convolutional layer, batch normalization, and ReLU
    """
    def __init__(self, ch_in, ch_out, conv_type: str = "standard"):
        """
        Initialize the block

        Args:
            ch_in: number of input channels
            ch_out: number of output channels
            conv_type: one of "standard", "coord", "antialiased", or "3d"
        """
        super(single_conv, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.conv = nn.Sequential(
            conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, conv_type: str = "standard"):
        super(Attention_block, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.W_g = nn.Sequential(
            conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
