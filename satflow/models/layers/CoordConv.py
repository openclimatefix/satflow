"""A 2D convolution over an input tensor and coordinates on a grid"""
import torch
import torch.nn as nn


class AddCoords(nn.Module):
    """"Add input tensors for x and y dimensions that are evenly spaced coordinates from -1 to 1"""
    def __init__(self, with_r=False):
        """
        Initialize module
        
        Args:
            with_r: also add an input that is the distance from the center (polar coordinates)
                Default if False
        """
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Compute the forward pass

        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """A 2D convolution over an input tensor and coordinates on a grid"""
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        """
        A 2D convolution over an input tensor and coordinates on a grid

        Args:
            in_chanels: number of input channels
            out_channels: number of output channels
            with_r: also add an input that is the distance from the center (polar coordinates)
                Default if False
        """
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        """
        Compute the forward pass

        Args:
            x: shape(batch, channel, x_dim, y_dim)
        """
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
