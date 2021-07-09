import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def _unbind_images(x, dim=1):
    "only unstack images"
    if isinstance(x, torch.Tensor):
        if len(x.shape) >= 4:
            return x.unbind(dim=dim)
    return x


class StackUnstack(nn.Module):
    "Stack together inputs, apply module, unstack output"

    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.module = module

    @staticmethod
    def unbind_images(x, dim=1):
        return _unbind_images(x, dim)

    def forward(self, *args):
        inputs = [torch.stack(x, dim=self.dim) for x in args]
        outputs = self.module(*inputs)
        if isinstance(outputs, (tuple, list)):
            return [self.unbind_images(output, dim=self.dim) for output in outputs]
        else:
            return outputs.unbind(dim=self.dim)


def StackLoss(loss_func=F.mse_loss, axis=-1):
    def _inner_loss(x, y):
        x = torch.cat(x, axis)
        y = torch.cat(y, axis)
        return loss_func(x, y)

    return _inner_loss


class MultiImageDice:
    "Dice coefficient metric for binary target in segmentation"

    def __init__(self, axis=1):
        self.axis = axis

    def reset(self):
        self.inter, self.union = 0, 0

    def accumulate(self, pred, y):
        x = torch.cat(pred, -1)
        y = torch.cat(y, -1)
        pred = x.argmax(dim=self.axis).flatten()
        targ = np.flatten(y)
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()

    @property
    def value(self):
        return 2.0 * self.inter / self.union if self.union > 0 else None
