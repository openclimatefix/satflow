import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=96):
        super().__init__()

        self.in_channel = in_channel
        self.bn = nn.BatchNorm2d(self.in_channel, affine=False)

        self.embed = nn.Linear(n_condition, self.in_channel * 2)
        self.embed.weight.data[:, : self.in_channel].normal_(1, 0.02)
        self.embed.weight.data[:, self.in_channel :].zero_()

    def forward(self, x, class_id):
        out = self.bn(x)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        # gamma = gamma.unsqueeze(2).unsqueeze(3)
        # beta = beta.unsqueeze(2).unsqueeze(3)
        gamma = gamma.view(-1, self.in_channel, 1, 1)
        beta = beta.view(-1, self.in_channel, 1, 1)
        out = gamma * out + beta

        return out


if __name__ == "__main__":

    cn = ConditionalNorm(3, 2)
    x = torch.rand([4, 3, 64, 64])
    class_id = torch.rand([4, 2])
    y = cn(x, class_id)
    print(cn)
    print(x.size())
    print(y.size())
