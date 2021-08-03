import functools

import torch
from torch import nn as nn
from torch.nn.modules.pixelshuffle import PixelShuffle
from torch.nn.utils.parametrizations import spectral_norm
from typing import Union, Tuple, List
from satflow.models.gan.common import get_norm_layer, init_net, GBlock
from satflow.models.utils import get_conv_layer
from satflow.models.layers import ConvGRU
import antialiased_cnns


def define_generator(
    input_nc,
    output_nc,
    ngf,
    netG: Union[str, torch.nn.Module],
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if isinstance(netG, torch.nn.Module):
        net = netG
    elif netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9
        )
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6
        )
    elif netG == "unet_128":
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    elif netG == "unet_256":
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(net, init_type, init_gain)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        conv_type: str = "standard",
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        conv2d = get_conv_layer(conv_type)
        model = [
            nn.ReflectionPad2d(3),
            conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if conv_type == "antialiased":
                block = [
                    conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    antialiased_cnns.BlurPool(ngf * mult * 2, stride=2),
                ]
            else:
                block = [
                    conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]

            model += block

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self, dim, padding_type, norm_layer, use_dropout, use_bias, conv_type: str = "standard"
    ):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        conv2d = get_conv_layer(conv_type)
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, conv2d
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias, conv2d: torch.nn.Module
    ):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        conv_type: str = "standard",
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            conv_type=conv_type,
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                conv_type=conv_type,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv_type=conv_type,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv_type=conv_type,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            conv_type=conv_type,
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            conv_type=conv_type,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        conv_type: str = "standard",
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        conv2d = get_conv_layer(conv_type)
        if conv_type == "antialiased":
            antialiased = True
            downconv = conv2d(
                input_nc, inner_nc, kernel_size=4, stride=1, padding=1, bias=use_bias
            )
            blurpool = antialiased_cnns.BlurPool(inner_nc, stride=2)
        else:
            antialiased = False
            downconv = conv2d(
                input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv, blurpool] if antialiased else [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = (
                [downrelu, downconv, downnorm, blurpool]
                if antialiased
                else [downrelu, downconv, downnorm]
            )
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NowcastingSampler(torch.nn.Module):
    def __init__(
        self,
        forecast_steps: int = 18,
        latent_channels: int = 768,
        context_channels: int = 384,
        output_channels: int = 1,
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.
        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.convGRU1 = ConvGRU(
            input_dim=latent_channels, hidden_dim=context_channels, kernel_size=(3, 3), n_layers=1
        )
        print(f" G1 Latent Input: {latent_channels} Output: {latent_channels // 2}")
        self.g1 = GBlock(input_channels=latent_channels, output_channels=latent_channels // 2)
        self.convGRU2 = ConvGRU(
            input_dim=latent_channels // 2,
            hidden_dim=context_channels // 2,
            kernel_size=(3, 3),
            n_layers=1,
        )
        self.g2 = GBlock(input_channels=latent_channels // 2, output_channels=latent_channels // 4)
        self.convGRU3 = ConvGRU(
            input_dim=latent_channels // 4,
            hidden_dim=context_channels // 4,
            kernel_size=(3, 3),
            n_layers=1,
        )
        self.g3 = GBlock(input_channels=latent_channels // 4, output_channels=latent_channels // 8)
        self.convGRU4 = ConvGRU(
            input_dim=latent_channels // 8,
            hidden_dim=context_channels // 8,
            kernel_size=(3, 3),
            n_layers=1,
        )
        self.g4 = GBlock(
            input_channels=latent_channels // 8, output_channels=latent_channels // 16
        )
        self.bn = torch.nn.BatchNorm2d(latent_channels // 16)
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=latent_channels // 16, out_channels=4 * output_channels, kernel_size=1
            )
        )
        self.depth2space = PixelShuffle(upscale_factor=2)

        # Now make copies of the entire stack, one for each future timestep
        stacks = torch.nn.ModuleDict()
        for i in range(forecast_steps):
            stacks[f"forecast_{i}"] = torch.nn.ModuleList(
                [
                    self.convGRU1,
                    self.g1,
                    self.convGRU2,
                    self.g2,
                    self.convGRU3,
                    self.g3,
                    self.convGRU4,
                    self.g4,
                    self.bn,
                    self.relu,
                    self.conv_1x1,
                    self.depth2space,
                ]
            )
        self.stacks = stacks

    def forward(
        self, conditioning_states: List[torch.Tensor], latent_dim: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one
        forecasts = []
        init_states = [torch.unsqueeze(c, dim=1) for c in conditioning_states]
        # Need to expand latent dim to the batch size
        # latent_dim = torch.cat(init_states[0].size()[0]*[latent_dim])
        latent_dim = torch.unsqueeze(latent_dim, dim=0)
        for i in range(self.forecast_steps):
            # Start at lowest one and go up, conditioning states
            # ConvGRU1
            # print(latent_dim.shape)
            # print(init_states[3].shape)
            x, recurrent_state = self.stacks[f"forecast_{i}"][0](
                latent_dim, hidden_state=init_states[3]
            )
            # Update for next timestep
            init_states[3] = recurrent_state
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            print(
                f"Latent Dim: {latent_dim.shape}, X: {x.shape}, Reccurent: {recurrent_state.shape}"
            )
            # GBlock1
            x = self.stacks[f"forecast_{i}"][1](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=0)
            # ConvGRU2
            x, recurrent_state = self.stacks[f"forecast_{i}"][2](x, hidden_state=init_states[2])
            # Update for next timestep
            init_states[2] = recurrent_state
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock2
            x = self.stacks[f"forecast_{i}"][3](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=0)
            # ConvGRU3
            x, recurrent_state = self.stacks[f"forecast_{i}"][4](x, hidden_state=init_states[1])
            # Update for next timestep
            init_states[1] = recurrent_state
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock3
            x = self.stacks[f"forecast_{i}"][5](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=0)
            # ConvGRU4
            x, recurrent_state = self.stacks[f"forecast_{i}"][6](x, hidden_state=init_states[0])
            # Update for next timestep
            init_states[0] = recurrent_state
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock4
            x = self.stacks[f"forecast_{i}"][7](x)
            # BN
            x = self.stacks[f"forecast_{i}"][8](x)
            # ReLU
            x = self.stacks[f"forecast_{i}"][9](x)
            # Conv 1x1
            x = self.stacks[f"forecast_{i}"][10](x)
            # Depth2Space
            x = self.stacks[f"forecast_{i}"][11](x)
            forecasts.append(x)
        return forecasts
