from functools import lru_cache
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import torch.distributions as distributions


limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor


def config_filename(height, width, kernel_size, padding, stride, cuda):
    str_ = f"{height}-{width}-{kernel_size}-{padding}-{stride}-{cuda}.dat"
    return str_

class LatencyTable(nn.Module):

    def __init__(self, filename, bias=-1, multiplier=10):
        super().__init__()
        with open(filename) as f:
            lines = f.readlines()
        data = []
        for line in lines:
            i, j, lat = line.split(",")
            i = (int(i) + bias) // multiplier
            j = (int(j) + bias) // multiplier
            lat = float(lat)
            data.append((i, j, lat))
        self.bias = bias
        self.multiplier = multiplier
        row_size, col_size = i + 1, j + 1
        self.register_buffer("table", torch.empty(row_size, col_size))
        for i, j, lat in data:
            self.table[i, j] = lat
        self.filename = filename

    @classmethod
    @lru_cache(maxsize=10000)
    def find(cls, height, width, conv, bias=-1, multiplier=10, cuda=False, device=None):
        cfg_filename = config_filename(height, width, conv.kernel_size[0], conv.padding[0], conv.stride[0], cuda)
        lat_table = cls(cfg_filename, bias=bias, multiplier=multiplier)
        if device is not None:
            lat_table = lat_table.to(device)
        # print(lat_table.table[:2, :2].cpu().numpy())
        # print(lat_table(torch.cuda.FloatTensor([10.9999]), torch.cuda.FloatTensor([10.9999])).cpu().numpy())
        # print(lat_table(torch.cuda.FloatTensor([1]), torch.cuda.FloatTensor([1])).cpu().numpy())
        # print("=" * 100)
        return lat_table

    def __str__(self):
        return self.filename

    def forward(self, i, j):
        i = ((i + self.bias).float() / self.multiplier)
        j = ((j + self.bias).float() / self.multiplier)
        i_f = i.floor().long()
        j_f = j.floor().long()
        i_c = (i + 1E-6).ceil().long()
        j_c = (j + 1E-6).ceil().long()
        i_df = (i - i_f.float()).abs_()
        j_df = (j - j_f.float()).abs_()
        i_dc = (i - i_c.float()).abs_()
        j_dc = (j - j_c.float()).abs_()
        o = self.table[i_f, j_f]
        r = self.table[i_c, j_f]
        t = self.table[i_f, j_c]
        rt = self.table[i_c, j_c]
        return j_dc * (i_dc * o + i_df * r) + j_df * (i_dc * t + i_df * rt)


class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, in_features, out_features, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        self.after = False
        self.before = False
        self.mask = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_dist(self):
        dist = distributions.bernoulli.Bernoulli(1 - self.cdf_qz(0))
        return dist

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def reset(self):
        self.mask = None
        self.mask_bias = None
        self.mask_weights = None
        self.after = False
        self.before = False

    def prune_mask(self, mask=None, reverse=False, conv=None):
        before = not reverse
        if mask is not None:
            self.mask = mask
            if before:
                self.before = True
                if self.after:
                    self.mask_weights = self.mask_weights[self.mask]
                else:
                    self.mask_weights = self.weights[self.mask]
                if self.use_bias:
                    self.mask_bias = self.bias
            else:
                self.after = True
                if self.before:
                    self.mask_weights = self.mask_weights[:, self.mask]
                    if self.use_bias:
                        self.mask_bias = self.mask_bias[self.mask]
                else:
                    self.mask_weights = self.weights[:, self.mask]
                    if self.use_bias:
                        self.mask_bias = self.bias[self.mask]
        elif conv is not None:
            weights = self.mask_weights if self.after else self.weights
            neuron_size = weights.size(0) // conv.size(0)
            conv = conv.unsqueeze(-1).expand(-1, neuron_size).contiguous().view(-1)
            self.mask_weights = weights[conv]
            self.mask_bias = self.bias.data
            self.mask = conv

    def n_active(self):
        pi = F.sigmoid(self.qz_loga)
        pi = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        return (pi != 0).long().sum()

    def forward(self, input):
        if self.local_rep or not self.training:
            if self.mask is None:
                z = self.sample_z(input.size(0), sample=self.training)
                xin = input.mul(z)
                output = xin.mm(self.weights)
            else:
                output = input.mm(self.mask_weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            if self.mask is None:
                output.add_(self.bias)
            else:
                output.add_(self.mask_bias)
        return output

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 droprate_init=0.5, temperature=2./3., weight_decay=1., lamba=1., local_rep=False, load_lat_table=True, 
                 nodrop=False, **kwargs):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.nodrop = nodrop
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.temperature = temperature
        self.use_bias = False
        self.weights = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.qz_loga = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep
        self.mask = None
        self.load_lat_table = load_lat_table
        self.after = False
        self.before = False
        self.index_mask = None
        self.frozen = False

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        print(self)

    def freeze(self):
        self.frozen = True

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode='fan_in')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def compute_pi(self):
        pi = F.sigmoid(self.qz_loga)
        pi = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        return pi

    def n_active(self):
        pi = self.compute_pi()
        return (pi != 0).long().sum()

    def sample_dist(self):
        dist = distributions.bernoulli.Bernoulli(1 - self.cdf_qz(0))
        return dist

    def reset(self):
        self.mask = None
        self.mask_weights = None
        self.mask_bias = None
        self.after = False
        self.before = False

    def index_mask(self, mask):
        self.index_mask = mask

    def prune_mask(self, mask, reverse=False):
        self.mask = mask
        before = reverse
        if before:
            if self.after:
                self.mask_weights = self.mask_weights[:, self.mask]
            else:
                self.mask_weights = self.weights[:, self.mask]
                if self.use_bias:
                    self.mask_bias = self.bias.data
            self.before = True
        else:
            if self.before:
                self.mask_weights = self.mask_weights[self.mask]
                if self.use_bias:
                    self.mask_bias = self.mask_bias[self.mask]
            else:
                self.mask_weights = self.weights[self.mask]
                if self.use_bias:
                    self.mask_bias = self.bias[self.mask]
            self.after = True

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def tie_gates(self, l0_layer):
        self.qz_loga = l0_layer.qz_loga

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 3).sum(2).sum(1)
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum((1 - q0) * (.5 * self.prior_prec * self.bias.pow(2) -
                                                                    self.lamba))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        ppos = torch.sum(1 - self.cdf_qz(0))
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(floatTensor(batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            if self.frozen:
                return (pi * (limit_b - limit_a) + limit_a).clamp(0, 1)
            else:
                return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        if self.load_lat_table:
            # print(self.in_channels, self.out_channels, self.kernel_size, input_.size(-2), input_.size(-1), self.stride, self.padding)
            self.lat_table = LatencyTable.find(input_.size(-2), input_.size(-1), self, device=self.weights.device)
            self.load_lat_table = False
        b = None if not self.use_bias else self.bias
        if self.local_rep or not self.training or self.frozen:
            if self.mask is None:
                output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation, self.groups)
                z = self.sample_z(output.size(0), sample=self.training and not self.frozen)
                if self.frozen and not self.nodrop:
                    return F.dropout(output.mul(z), self.droprate_init)
                else:
                    return output.mul(z)
            else:
                output = F.conv2d(input_, self.mask_weights, self.mask_bias, self.stride, self.padding, self.dilation, self.groups)
                return output
        else:
            weights = self.sample_weights()
            output = F.conv2d(input_, weights, None, self.stride, self.padding, self.dilation, self.groups)
            return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, '
             'droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, '
             'lamba={lamba}, local_rep={local_rep}, load_lat_table={load_lat_table}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)






