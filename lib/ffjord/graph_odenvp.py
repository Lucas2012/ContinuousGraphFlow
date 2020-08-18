import torch
import torch.nn as nn
import lib.ffjord.layers as layers
from lib.ffjord.layers.odefunc import ODEGraphnet
import numpy as np


class GraphODENVP(nn.Module):
    """
    Graphical Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
        self,
        input_size,
        strides,
        n_scale=float('inf'),
        threshold_output_size=2,
        n_blocks=2,
        intermediate_dims=(32,),
        nonlinearity="softplus",
        squash_input=True,
        alpha=0.05,
        unit_type="conv",
        mp_type="generic",
        num_graph_layers=1,
        conv=True, 
        ifgate=False, 
        cnf_kwargs=None,
        network_choice=None,
        prior_config=[False, 0],
        conv_embed_config=[False, 0, 0]
    ):
        super(GraphODENVP, self).__init__()
        self.n_scale = min(n_scale, self._calc_n_scale(input_size, threshold_output_size))
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.strides = strides
        self.nonlinearity = nonlinearity
        self.squash_input = squash_input
        self.alpha = alpha
        self.ifconv = conv
        self.unit_type = unit_type
        self.mp_type = mp_type
        self.num_graph_layers=num_graph_layers
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}
        self.network_choice = network_choice
        self.require_label = prior_config[0]
        self.conv_embed_config = conv_embed_config
        self.num_class = prior_config[1]
        self.ifgate = ifgate
        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[2:] for o in self.calc_output_size(input_size)]

    def set_graphs(self, graphs):
        for l in self.transforms:
          l.set_graphs(graphs)

    def set_embed_info(self, embed_info):
        for l in self.transforms:
          l.set_embed_info(embed_info)

    def set_E(self, E, shape=None):
        for l in self.transforms:
          l.set_E(E, shape)

    def set_masks(self, masks):
        for l in self.transforms:
          l.set_masks(masks)

    def _build_net(self, input_size):
        _, _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            prior_config=[self.require_label, self.num_class, h*w*c//2]
            if i == self.n_scale - 1:
              prior_config[-1] = h*w*c
            transforms.append(
                GraphStackedCNFLayers(
                    initial_size=(c, h, w),
                    idims=self.intermediate_dims,
                    strides=self.strides,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform())
                    if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    unit_type = self.unit_type,
                    mp_type=self.mp_type,
                    num_graph_layers = self.num_graph_layers,
                    conv=self.ifconv,
                    ifgate=self.ifgate,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                    network_choice=self.network_choice,
                    prior_config=prior_config,
                    conv_embed_config=self.conv_embed_config,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def get_regularization(self):
        if len(self.regularization_fns) == 0:
            return None

        acc_reg_states = tuple([0.] * len(self.regularization_fns))
        for module in self.modules():
            if isinstance(module, layers.CNF):
                acc_reg_states = tuple(
                    acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states())
                )
        return sum(state * coeff for state, coeff in zip(acc_reg_states, self.regularization_coeffs))

    def _calc_n_scale(self, input_size, threshold_output_size):
        _, _, _, h, w = input_size
        n_scale = 0
        while h >= threshold_output_size and w >= threshold_output_size:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, k, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, k, c, h, w))
            else:
                output_sizes.append((n, k, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, reverse=False,temperature=0.7):
        if reverse:
            return self._generate(x, logpx,temperature=temperature)
        else:
            return self._logdensity(x, logpx)

    def _logdensity(self, x, logpx=None):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx = self.transforms[idx].forward(x, _logpx)
            if idx < len(self.transforms) - 1:
                d = x.size(2) // 2
                x, factor_out = x[:, :, :d], x[:, :, d:]
            else:
                factor_out = x
            out.append(factor_out)
        out = [o.contiguous().view(o.size()[0], o.size()[1], -1) for o in out]
        return out if logpx is None else (out, _logpx)

    def _generate(self, zs,logpz=None,temperature=0.7):
        zs = [_z.view(_z.size()[0], _z.size()[1], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz = self.transforms[-1](zs[-1] * temperature, _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=2)
            z_prev, _logpz = self.transforms[idx](z_prev, _logpz, reverse=True)
        return z_prev if logpz is None else (z_prev, _logpz)

    def get_logp(self, z, label):
        logp = 0
        for idx in range(len(self.transforms)):
          logp = self.transforms[idx].get_logp(z[idx], label).sum(1).sum(1, keepdim=True) + logp

        return logp

    def get_prior_samples(self, label):
        zs = []
        for idx in range(len(self.transforms)):
          zs += [self.transforms[idx].get_prior_samples(label)]

        return zs


class GraphStackedCNFLayers(layers.SequentialFlow):
    def __init__(
        self,
        initial_size,
        idims=(32,),
        strides=None,
        nonlinearity="softplus",
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        unit_type="conv",
        mp_type="generic",
        num_graph_layers = 1,
        conv=True,
        ifgate=False,
        cnf_kwargs={},
        network_choice=None,
        prior_config=None,
        conv_embed_config=None,
    ):
        self.unit_type = unit_type 
        self.ifconv = conv
        self.prior_config = prior_config
        self.conv_embed_config = conv_embed_config
        self.num_graph_layers=num_graph_layers
        self.mp_type=mp_type
        self.ifgate=ifgate
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size, network_choice):
            if self.unit_type == "conv" and self.mp_type=="generic":
              net = network_choice(idims, size, strides, self.ifconv, layer_type="concat", nonlinearity=nonlinearity, ifgate=self.ifgate,num_func=5, conv_embed_config=conv_embed_config)
              f = layers.ODEfunc(net)
            
            elif self.unit_type == "linear" and self.mp_type=="generic":
              net = network_choice(idims, size, strides, self.ifconv, layer_type="concat", nonlinearity=nonlinearity, num_func=5,reshape=True)
              f = layers.ODEfunc(net)

            elif self.unit_type == "ae" and self.mp_type=="generic":
              net = network_choice(idims, size, strides, self.ifconv, layer_type="concat", nonlinearity=nonlinearity, num_graph_layers=self.num_graph_layers)
              f= layers.GraphAutoencoderODEfunc(net)
            
            elif self.unit_type == "conv" and self.mp_type=="affine":
              net = network_choice(idims, size, strides, self.ifconv, layer_type="concat", nonlinearity=nonlinearity,reshape=True)
              f = layers.ODEAffineGraphfunc(net)

            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [layers.CNF(_make_odefunc(initial_size, network_choice), **cnf_kwargs) for _ in range(n_blocks)]
            post = [layers.CNF(_make_odefunc(after_squeeze_size, network_choice), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [layers.SqueezeLayer(2)] + post
        else:
            chain += [layers.CNF(_make_odefunc(initial_size, network_choice), **cnf_kwargs) for _ in range(n_blocks)]

        super(GraphStackedCNFLayers, self).__init__(chain, prior_config)

