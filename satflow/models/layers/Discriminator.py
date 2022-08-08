import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from satflow.models.layers.Normalization import ConditionalNorm, SpectralNorm

# from Module.Attention import SelfAttention
# from Module.GResBlock import GResBlock


########################################################################################
class Spectral_Norm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = Spectral_Norm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + "_u", u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + "_u", u)


def spectral_norm(module, name="weight"):
    Spectral_Norm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out


class GBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        bn=True,
        activation=F.relu,
        upsample=True,
        downsample=False,
    ):
        super().__init__()

        gain = 2**0.5

        self.conv0 = SpectralNorm(
            nn.Conv2d(
                in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True
            )
        )
        self.conv1 = SpectralNorm(
            nn.Conv2d(
                out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True
            )
        )

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class SpatialDiscriminator(nn.Module):
    def __init__(self, chn=128, n_class=4):
        super().__init__()

        self.pre_conv = nn.Sequential(
            SpectralNorm(
                nn.Conv2d(3, 2 * chn, 3, padding=1),
            ),
            nn.ReLU(),
            SpectralNorm(
                nn.Conv2d(2 * chn, 2 * chn, 3, padding=1),
            ),
            nn.AvgPool2d(2),
        )
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 2 * chn, 1))

        self.conv1 = GBlock(2 * chn, 4 * chn, bn=False, upsample=False, downsample=True)
        self.attn = SelfAttention(4 * chn)
        self.conv2 = nn.Sequential(
            GBlock(4 * chn, 8 * chn, bn=False, upsample=False, downsample=True),
            GBlock(8 * chn, 16 * chn, bn=False, upsample=False, downsample=True),
            GBlock(16 * chn, 16 * chn, bn=False, upsample=False, downsample=True),
        )

        self.linear = SpectralNorm(nn.Linear(16 * chn, 1))

        self.embed = nn.Embedding(n_class, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = SpectralNorm(self.embed)

    def forward(self, x, class_id):
        # reshape input tensor from BxTxCxHxW to BTxCxHxW
        batch_size, T, C, W, H = x.size()

        x = x.view(batch_size * T, C, H, W)

        out = self.pre_conv(x)
        out = out + self.pre_skip(F.avg_pool2d(x, 2))

        # reshape back to B x T x C x H x W

        # out = out.view(batch_size, T, -1, H // 2, W // 2)

        out = self.conv1(out)  # B x T x C x H x W
        # out = out.permute(0, 2, 1, 3, 4) # B x C x T x H x W

        out = self.attn(out)  # B x C x T x H x W
        # out = out.permute(0, 2, 1, 3, 4).contiguous() # B x T x C x H x W

        out = self.conv2(out)

        out = F.relu(out)

        # out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.view(batch_size, T, out.size(2), -1) # B x T x C x H x W

        # sum on H and W axis
        out = out.sum(2)
        # sum on T axis
        # out = out.sum(1)

        out_linear = self.linear(out).squeeze(1)

        # repeat class_id for each frame
        # TODO: test in case multi-class
        class_id = class_id.view(-1, 1).repeat(1, T).view(-1)

        embed = self.embed(class_id)

        prod = (out * embed).sum(1)

        # out_linear = out_linear.view(-1, T)
        # prod = prod.view(-1, T)

        # score = (out_linear + prod).sum(1)

        return out_linear + prod


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Res3dBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3, 3],
        padding=1,
        stride=1,
        n_class=None,
        bn=True,
        activation=F.relu,
        upsample=True,
        downsample=False,
    ):
        super().__init__()

        gain = 2**0.5

        self.conv0 = SpectralNorm(
            nn.Conv3d(
                in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True
            )
        )
        self.conv1 = SpectralNorm(
            nn.Conv3d(
                out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True
            )
        )

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv3d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool3d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool3d(skip, 2)

        else:
            skip = input

        return out + skip


class TemporalDiscriminator(nn.Module):
    def __init__(self, chn=128, n_class=4):
        super().__init__()

        gain = 2**0.5

        self.pre_conv = nn.Sequential(
            SpectralNorm(nn.Conv3d(3, 2 * chn, 3, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv3d(2 * chn, 2 * chn, 3, padding=1)),
            nn.AvgPool3d(2),
        )
        self.pre_skip = SpectralNorm(nn.Conv3d(3, 2 * chn, 1))

        self.res3d = Res3dBlock(2 * chn, 4 * chn, bn=False, upsample=False, downsample=True)

        self.self_attn = SelfAttention(4 * chn)

        self.conv = nn.Sequential(
            GBlock(4 * chn, 8 * chn, bn=False, upsample=False, downsample=True),
            GBlock(8 * chn, 16 * chn, bn=False, upsample=False, downsample=True),
            GBlock(16 * chn, 16 * chn, bn=False, upsample=False, downsample=True),
        )

        self.linear = SpectralNorm(nn.Linear(16 * chn, 1))

        self.embed = nn.Embedding(n_class, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = SpectralNorm(self.embed)

    def forward(self, x, class_id):
        # pre-process with avg_pool2d to reduce tensor size
        # B, T, C, H, W = x.size()
        # x = F.avg_pool2d(x.view(B * T, C, H, W), kernel_size=2)
        # _, _, H, W = x.size()
        # x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # B x C x T x W x H

        out = self.pre_conv(x)
        out = out + self.pre_skip(F.avg_pool3d(x, 2))
        out = self.res3d(out)  # B x C x T x W x H

        # reshape to BTxCxWxH
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, W, H = out.size()
        out = out.view(B * T, C, W, H)

        out = self.self_attn(out)
        # out = out.permute(0, 2, 1, 3, 4).contiguous() # B x T x C x W x H

        out = self.conv(out)
        out = F.relu(out)

        # out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.view(batch_size, T, out.size(2), -1) # B x T x C x H x W

        # sum on H and W axis
        out = out.sum(2)
        # sum on T axis
        # out = out.sum(1)
        out_linear = self.linear(out).squeeze(1)

        # repeat class_id for each frame
        # TODO: test in case multi-class
        class_id = class_id.view(-1, 1).repeat(1, T).view(-1)

        embed = self.embed(class_id)

        prod = (out * embed).sum(1)

        # out_linear = out_linear.view(-1, T)
        # prod = prod.view(-1, T)

        # score = (out_linear + prod).sum(1)

        return out_linear + prod


########################################################################################


# class SpatialDiscriminator(nn.Module):

#     def __init__(self, chn=128, n_class=4):
#         super().__init__()

#         self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 2*chn, 3, padding=1), ),
#                                       nn.ReLU(),
#                                       SpectralNorm(nn.Conv2d(2*chn, 2*chn, 3, padding=1), ),
#                                       nn.AvgPool2d(2))
#         self.pre_skip = SpectralNorm(nn.Conv2d(3, 2*chn, 1))

#         self.conv1 = GResBlock(2*chn, 4*chn, bn=False, downsample_factor=2)
#         self.attn = SelfAttention(4*chn)
#         self.conv2 = nn.Sequential(
#             GResBlock(4*chn, 8*chn, bn=False, downsample_factor=2),
#             GResBlock(8*chn, 16*chn, bn=False, downsample_factor=2),
#             GResBlock(16*chn, 16*chn, bn=False, downsample_factor=2)
#         )

#         self.linear = SpectralNorm(nn.Linear(16*chn, 1))

#         self.embed = nn.Embedding(n_class, 16*chn)
#         self.embed.weight.data.uniform_(-0.1, 0.1)
#         self.embed = SpectralNorm(self.embed)

#     def forward(self, x, class_id):
#         # reshape input tensor from BxTxCxHxW to BTxCxHxW
#         batch_size, T, C, W, H = x.size()

#         x = x.view(batch_size * T, C, H, W)

#         out = self.pre_conv(x)
#         out = out + self.pre_skip(F.avg_pool2d(x, 2))

#         # reshape back to B x T x C x H x W

#         out = out.view(batch_size, T, -1, H // 2, W // 2)

#         out = self.conv1(out) # B x T x C x H x W
#         out = out.permute(0, 2, 1, 3, 4) # B x C x T x H x W

#         out = self.attn(out) # B x C x T x H x W
#         out = out.permute(0, 2, 1, 3, 4).contiguous() # B x T x C x H x W

#         out = self.conv2(out)

#         out = F.relu(out)

#         out = out.permute(0, 2, 1, 3, 4).contiguous()
#         out = out.view(out.size(0), out.size(1), -1)
#         # out = out.view(batch_size, T, out.size(2), -1) # B x T x C x H x W

#         # sum on H and W axis
#         out = out.sum(2)
#         # sum on T axis
#         # out = out.sum(1)

#         out_linear = self.linear(out).squeeze(1)

#         embed = self.embed(class_id)

#         prod = (out * embed).sum(1)

#         return out_linear + prod


# def conv3x3x3(in_planes, out_planes, stride=1):
#     # 3x3x3 convolution with padding
#     return nn.Conv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         bias=False)


# class Res3dBlock(nn.Module):

#     def __init__(self, in_channel, out_channel, stride=1, downsample=False):
#         super(Res3dBlock, self).__init__()
#         self.conv1 = conv3x3x3(in_channel, out_channel, stride)
#         self.bn1 = nn.BatchNorm3d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(out_channel, out_channel)
#         self.bn2 = nn.BatchNorm3d(out_channel)
#         self.downsample = downsample
#         if self.downsample:
#             self.conv_sc = nn.Sequential(
#                 nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm3d(out_channel)
#             )
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample:
#             residual = self.conv_sc(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class TemporalDiscriminator(nn.Module):

#     def __init__(self, chn=128, n_class=4):
#         super().__init__()

#         gain = 2 ** 0.5

#         self.pre_conv = nn.Sequential(
#             SpectralNorm(nn.Conv3d(3, 2*chn, 3, padding=1)),
#             nn.ReLU(),
#             SpectralNorm(nn.Conv3d(2*chn, 2*chn, 3, padding=1)),
#             nn.AvgPool3d(2)
#         )
#         self.pre_skip = SpectralNorm(nn.Conv3d(3, 2*chn, 1))

#         self.res3d = Res3dBlock(2*chn, 4*chn, downsample=True)

#         self.self_attn = SelfAttention(4*chn)

#         self.conv = nn.Sequential(
#             GResBlock(4*chn, 8*chn, bn=False, downsample_factor=2),
#             GResBlock(8*chn, 16*chn, bn=False, downsample_factor=2),
#             GResBlock(16*chn, 16*chn, bn=False, downsample_factor=2)
#         )

#         self.linear = SpectralNorm(nn.Linear(16*chn, 1))

#         # self.embed = nn.Embedding(n_class, 16*chn)
#         # self.embed.weight.data.uniform_(-0.1, 0.1)
#         # self.embed = SpectralNorm(self.embed)

#     def forward(self, x, class_id):
#         # pre-process with avg_pool2d to reduce tensor size
#         # B, T, C, H, W = x.size()
#         # x = F.avg_pool2d(x.view(B * T, C, H, W), kernel_size=2)
#         # _, _, H, W = x.size()
#         # x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # B x C x T x W x H

#         out = self.pre_conv(x)
#         out = out + self.pre_skip(F.avg_pool3d(x, 2))
#         out = self.res3d(out) # B x C x T x H x W

#         out = self.self_attn(out)
#         out = out.permute(0, 2, 1, 3, 4).contiguous() # B x T x C x W x H

#         out = self.conv(out)
#         out = F.relu(out)

#         out = out.permute(0, 2, 1, 3, 4).contiguous()
#         out = out.view(out.size(0), out.size(1), -1)
#         # out = out.view(batch_size, T, out.size(2), -1) # B x T x C x H x W

#         # sum on H and W axis
#         out = out.sum(2)
#         # sum on T axis
#         # out = out.sum(1)
#         out_linear = self.linear(out).squeeze(1)

#         # print(out_linear)

#         # embed = self.embed(class_id)

#         # prod = (out * embed).sum(1)

#         return out_linear #+ prod


if __name__ == "__main__":

    batch_size = 6
    n_frames = 8
    n_class = 4
    n_chn = 4

    model = TemporalDiscriminator(chn=n_chn, n_class=n_class)
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0, 0.9), weight_decay=0.00001
    )
    for i in range(100):
        data = torch.randn((batch_size, n_frames, 3, 64, 64)).cuda()

        label = torch.randint(0, n_class, (batch_size,)).cuda()
        # B, T, C, H, W = data.size()
        # data = F.avg_pool2d(data.view(B * T, C, H, W), kernel_size=2)
        # _, _, H, W = data.size()
        # data = data.view(B, T, C, H, W)

        # # transpose to BxCxTxHxW
        # data = data.transpose(1, 2).contiguous()

        out = model(data, label)
        loss = torch.mean(out)
        print(loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
