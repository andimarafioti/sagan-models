""" Code mixed from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
and issues"""

import torch
import torch.nn as nn


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, inChannels):
        super(Self_Attn, self).__init__()
        self.key = nn.Conv2d(inChannels, inChannels//8, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.key.weight)
        torch.nn.utils.spectral_norm(self.key)

        self.query = nn.Conv2d(inChannels, inChannels//8, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.query.weight)
        torch.nn.utils.spectral_norm(self.query)

        self.pool2d = nn.MaxPool2d(2, stride=2)

        self.value = nn.Conv2d(inChannels, inChannels//2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.value.weight)
        torch.nn.utils.spectral_norm(self.value)

        self.self_att = nn.Conv2d(inChannels//2, inChannels, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.self_att.weight)
        torch.nn.utils.spectral_norm(self.self_att)

        self.sigma = nn.Parameter(torch.tensor(0.0))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """
        batchsize, C, H, W = x.size()
        location_num = H * W  # Number of features
        theta = self.key(x).view(batchsize, C//8, location_num)    # Keys
        phi = self.pool2d(self.query(x)).view(batchsize, C//8, location_num//4)  # Queries

        attn = torch.bmm(theta.permute(0, 2, 1), phi)  # Scores                [B, N, N//4]
        attn = self.softmax(attn)  # Attention Map         [B, N, N//4]

        g = self.pool2d(self.value(x)).view(batchsize, C//2, location_num//4)  # Values [B, C_bar, N//4]
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))  # Value x Softmax       [B, C_bar, N]
        attn_g = attn_g.view(batchsize, C//2, H, W)  # Recover image shape
        attn_g = self.self_att(attn_g)  # Self-Attention output [B, C, H, W]

        y = x + self.sigma * attn_g  # Learnable sigma + residual
        return y, attn


""" Code base from https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/sn_projection_cgan_64x64_143c.ipynb"""


class ResGenBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=torch.nn.functional.relu, upsample=False, n_classes=0):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = torch.nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.utils.spectral_norm(self.c1)
        self.c2 = torch.nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.utils.spectral_norm(self.c2)
        if n_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm(in_channels, n_classes)
            self.b2 = CategoricalConditionalBatchNorm(hidden_channels, n_classes)
        else:
            self.b1 = BatchNorm2d(in_channels)
            self.b2 = BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)
            torch.nn.utils.spectral_norm(self.c_sc)

    def forward(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        if self.upsample:
            h = torch.nn.functional.interpolate(h, scale_factor=2)
        h = self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.learnable_sc:
            if self.upsample:
                x = torch.nn.functional.interpolate(x, scale_factor=2)
            sc = self.c_sc(x)
        else:
            sc = x
        return h + sc


class ResNetGenerator(torch.nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=torch.nn.functional.relu, n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = torch.nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.utils.spectral_norm(self.l1)
        self.block1 = ResGenBlock(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
        self.block2 = ResGenBlock(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = ResGenBlock(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.attn = Self_Attn(ch * 4)
        self.block4 = ResGenBlock(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = ResGenBlock(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b6 = BatchNorm2d(ch)
        self.l6 = torch.nn.Conv2d(ch, 3, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l6.weight)
        torch.nn.init.zeros_(self.l6.bias)
        torch.nn.utils.spectral_norm(self.l6)

    def forward(self, batchsize=64, z=None, y=None):
        if z is None:
            z = torch.randn(batchsize, self.dim_z).cuda()
        if y is None and self.n_classes > 0:
            y = torch.randint(0, self.n_classes, (batchsize,), dtype=torch.long).cuda()
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block1(h, y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h, p1 = self.attn(h)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.l6(h))
        return h, p1


class ResDisBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=torch.nn.functional.relu, downsample=False):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = torch.nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.utils.spectral_norm(self.c1)
        self.c2 = torch.nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)
            torch.nn.utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = self.activation(x)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = torch.nn.functional.avg_pool2d(h, 2)
        if self.learnable_sc:
            sc = self.c_sc(x)
            if self.downsample:
                sc = torch.nn.functional.avg_pool2d(sc, 2)
        else:
            sc = x
        return h + sc


class ResDisOptimizedBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=torch.nn.functional.relu):
        super().__init__()
        self.activation = activation
        self.c1 = torch.nn.Conv2d(in_channels, out_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.utils.spectral_norm(self.c1)
        self.c2 = torch.nn.Conv2d(out_channels, out_channels, ksize, padding=pad)
        torch.nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.utils.spectral_norm(self.c2)
        self.c_sc = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        torch.nn.init.xavier_uniform_(self.c_sc.weight)
        torch.nn.init.zeros_(self.c_sc.bias)
        torch.nn.utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = torch.nn.functional.avg_pool2d(h, 2)

        sc = torch.nn.functional.avg_pool2d(x, 2)
        sc = self.c_sc(sc)
        return h + sc


class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=torch.nn.functional.relu):
        super().__init__()
        self.activation = activation
        self.block1 = ResDisOptimizedBlock(3, ch)
        self.block2 = ResDisBlock(ch, ch * 2, activation=activation, downsample=True)
        self.attn = Self_Attn(ch * 2)
        self.block3 = ResDisBlock(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = ResDisBlock(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = ResDisBlock(ch * 8, ch * 16, activation=activation, downsample=True)
        self.block6 = ResDisBlock(ch * 16, ch * 16, activation=activation, downsample=False)
        self.l6 = torch.nn.Linear(ch * 16, 1)
        torch.nn.init.xavier_uniform_(self.l6.weight)
        torch.nn.init.zeros_(self.l6.bias)
        torch.nn.utils.spectral_norm(self.l6)

        if n_classes > 0:
            self.l_y = torch.nn.Embedding(n_classes, ch * 16)
            torch.nn.init.xavier_uniform_(self.l_y.weight)
            torch.nn.utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h, p1 = self.attn(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output, p1


class BatchNorm2d(torch.nn.BatchNorm2d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()


class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)
