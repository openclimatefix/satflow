import torch
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable

"""
Taken from https://github.com/vagr8/R_Unet
"""


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    ## initialize h and c internal state only, keep weight mask the same
    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(
                self.device
            )
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(
                self.device
            ),
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(
                self.device
            ),
        )


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(
        self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]
    ):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        for i in range(self.num_layers):
            name = "cell{}".format(i)
            cell = ConvLSTMCell(
                self.input_channels[i], self.hidden_channels[i], self.kernel_size
            )
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, init_token=False):

        outputs = []
        if init_token == True:
            self.internal_state = []

        for step in range(1):
            x = input

            for i in range(self.num_layers):
                name = "cell{}".format(i)

                # at first step of each video, pass init_token = True in, and initiallize internal states
                if step == 0 and init_token == True:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize,
                        hidden=self.hidden_channels[i],
                        shape=(height, width),
                    )
                    self.internal_state.append((h, c))

                # do forward
                (h, c) = self.internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                self.internal_state[i] = (x, new_c)

            # only record last output
            outputs.append(x)

        return outputs


if __name__ == "__main__":
    # gradient check
    convlstm = ConvLSTM(
        input_channels=1024,
        hidden_channels=[1024, 512, 512],
        kernel_size=3,
        step=3,
        effective_step=[2],
    )
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 1024, 16, 16))
    target = Variable(torch.randn(1, 512, 16, 16)).double()

    output = convlstm(input)
    output = output[0][0].double()
    print(np.asarray(output[0].detach(), dtype=float).shape)
    res = torch.autograd.gradcheck(
        loss_fn, (output, target), eps=1e-6, raise_exception=True
    )
    print(res)
