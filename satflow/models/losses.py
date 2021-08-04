import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssim_module = SSIM(**kwargs)

    def forward(self, X, Y):
        return 1.0 - self.ssim_module(X, Y)


class MS_SSIMLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, X, Y):
        return 1.0 - self.ssim_module(X, Y)


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

    """

    def __init__(self, weight_fn=None):
        super().__init__()
        self.weight_fn = weight_fn  # In Paper, weight_fn is max(y+1,24)

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value, assumes generated images are the mean predictions from
        6 calls to the generater (Monte Carlo estimation of the expectations for the latent variable)
        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            difference *= self.weight_fn(targets)
        difference /= targets.size(1) * targets.size(3) * targets.size(4)  # 1/HWN
        return torch.mean(
            torch.mean(torch.mean(difference, dim=-1), dim=-1), dim=1
        )  # Mean of Time/Width/Height


class NowcastingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, real_flag):
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


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


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
    if isinstance(loss, torch.nn.Module):
        return loss
    assert loss in [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
    ]
    if loss == "mse":
        criterion = F.mse_loss
    elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
        criterion = F.nll_loss
    elif loss in ["focal"]:
        criterion = FocalLoss()
    elif loss in ["ssim"]:
        criterion = SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ms_ssim"]:
        criterion = MS_SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["l1"]:
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion
