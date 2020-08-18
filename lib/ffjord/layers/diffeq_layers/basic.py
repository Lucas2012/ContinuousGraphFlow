import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, pi, exp,isnan
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class IgnoreLinear1x1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear1x1, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        x_shape = x.shape
        count   = x.nelement() / x.shape[-1]
        x       = x.view([int(count)] + list(x.shape)[-1:])
        return self._layer(x).view(x_shape)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[..., :1]) * t
        ttx = torch.cat([tt, x], -1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class HyperConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )


class IgnoreConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(IgnoreConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        return self._layer(x)


class IgnoreConv2d1x1(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, transpose=False):
        super(IgnoreConv2d1x1, self).__init__()
        ksize   = 1
        stride  = 1
        padding = 0
        dilation = 1
        groups  = 1
        module  = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        x_shape = x.shape
        count   = x.nelement() / x.shape[-1] / x.shape[-2] / x.shape[-3]
        x       = x.view([int(count)] + list(x.shape)[-3:])
        return self._layer(x).view(x_shape)

class SquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        self.stride = stride
        self.transpose = transpose
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(1, -1, 1, 1) \
            + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.stride = stride
        self._layer = module(
            dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(nn.Module):
    def __init__(
        self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class GraphLayerConvEmbed(nn.Module):
    def __init__(self, base_layer, params, activation_fns, embed_config=[0, 0]):
        super(GraphLayerConvEmbed, self).__init__()
        kwargs = params[2]
        module = nn.ConvTranspose2d if kwargs['transpose'] else nn.Conv2d
        self.conv_map = base_layer(params[0], params[1], **kwargs) 
        self.embed_class = embed_config[0]
        self.embed_dim   = embed_config[1]
 
        self.embed       = nn.Embedding(self.embed_class, self.embed_dim)

        self.unary_passing  = nn.Sequential(
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                activation_fns,
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding'])
                              )

        self.binary_passing = nn.Sequential(
                                module(2*params[1] + self.embed_dim, params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                activation_fns,
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding'])
                              )
        self.E = None
        self.masks = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E
 
    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        N, K, C, H, W = x.size()
        h = self.conv_map(t, x.view(N*K, C, H, W)).view(N, K, -1, H, W)
        C = h.size(2)
        if self.E is None:
          E_matrix = self._get_fc(N, K)
        else:
          E_matrix = self.E

        # h: [N, K, C, H, W]
        unary_message = self.unary_passing(h.view(N*K, C, H, W))
        unary_message = unary_message.view(N, K, C, H, W)
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C, H, W), \
                              h.unsqueeze(2).expand(N, K, K, C, H, W), \
                              self.embed(E_matrix.long()).view(N, K, K, -1, 1, 1).expand(N, K, K, -1, H, W)], dim=3)

        binary_message = self.binary_passing(h_matrix.view(N*K*K, -1, H, W)) * \
                         (E_matrix>0).float().view(N*K*K, 1, 1, 1).expand(N*K*K, C, H, W)

        E_pos = (E > 0).float()
        binary_message = binary_message.view(N, K, K, C, H, W).sum(1) / (1 + E_pos.sum(1).view(N, K, 1, 1, 1).expand(N, K, C, H, W))
        h = unary_message + binary_message

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1, 1, 1).expand(N, K, C, H, W)

        return h


class GraphLayerLinearEmbed(nn.Module):
    def __init__(self, base_layer, params, activation_fns, embed_config=[0, 0, 0]):
        super(GraphLayerLinearEmbed, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        self.embed_rel_class = embed_config[0]
        self.embed_obj_class = embed_config[1]
        self.embed_dim       = embed_config[2]
    
        self.embed_rel      = nn.Embedding(self.embed_rel_class, self.embed_dim)
        self.embed_obj      = nn.Embedding(self.embed_obj_class, self.embed_dim)

        self.unary_passing  = nn.Sequential(
                                nn.Linear(params[1] + self.embed_dim, params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )

        self.binary_passing = nn.Sequential(
                                nn.Linear(2*params[1] + self.embed_dim*3, params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )

        self.binary_passing_reverse = nn.Sequential(
                                        nn.Linear(2*params[1] + self.embed_dim*3, params[1]),
                                        activation_fns,
                                        nn.Linear(params[1], params[1])
                                      )

        self.E = None
        self.masks = None
        self.graphs = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E

    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E
 
    def set_graphs(self, graphs):
        self.graphs = graphs

    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        h   = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        if self.E is None:
          raise ValueError('In embed mode, edge matrix has to be specified.')
        else:
          E_matrix = self.E

        obj_embed = self.embed_obj(self.graphs)
        unary_message = self.unary_passing(torch.cat([h, obj_embed], dim=-1))
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                              h.unsqueeze(2).expand(N, K, K, C), \
                              self.embed_rel(E_matrix.long()), \
                              obj_embed.unsqueeze(1).expand(N, K, K, -1), \
                              obj_embed.unsqueeze(2).expand(N, K, K, -1)], dim=3)
        binary_message = self.binary_passing(h_matrix.view(N*K*K, -1)) * (E_matrix>0).float().view(N*K*K, 1).expand(N*K*K, C)
        binary_message_reverse = self.binary_passing_reverse(h_matrix.view(N*K*K, -1)) * (E_matrix>0).float().view(N*K*K, 1).expand(N*K*K, C)

        E_pos = (E_matrix > 0).float()
        binary_message = binary_message.view(N, K, K, C).sum(1) / (1 + E_pos.sum(1).view(N, K, 1).expand(N, K, C))
        binary_message_reverse = binary_message_reverse.view(N, K, K, C).sum(2) / (1 + E_pos.sum(2).view(N, K, 1).expand(N, K, C))
        h = unary_message + binary_message + binary_message_reverse

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)

        return h

class GraphLayerConv(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate, num_func=5, embed_config=[False, 0, 0]):
        super(GraphLayerConv, self).__init__()
        base_layer_kwargs = params[2]
        kwargs = params[3]
        module = nn.ConvTranspose2d if kwargs['transpose'] else nn.Conv2d
        self.conv_map = base_layer(params[0], params[1], **base_layer_kwargs) 

        self.embed_flag  = embed_config[0]
        self.embed_indim = embed_config[1]
        self.embed_dim   = 0
        self.ifgate = ifgate
        if ifgate:
            self.unary_gate     = nn.Sequential(
                                    module(params[1]+self.embed_dim, params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    activation_fns,
                                    module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    nn.Sigmoid()
                                  )


        if self.embed_flag:
          self.embed_dim   = embed_config[2]
          self.embedding = nn.Linear(self.embed_indim, self.embed_dim)

        self.unary_passing   = nn.Sequential(
                                 module(params[1] + self.embed_dim, params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                 activation_fns,
                                 module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding'])
                               )

        binary_passing_layers = [None]
        binary_gate_layers = [None]
        assert num_func==5

        for i in range(1, num_func):
            stride = kwargs['stride']
            assert stride == 1
            # up or down
            if i == 1 or i == 2:
              stride = [2, stride]
            # left or right
            elif i == 3 or i == 4:
              stride = [stride, 2]
            #if not self.embed_flag:
            binary_passing_layers += [nn.Sequential(
                                          module(params[1] + self.embed_dim, params[1], kwargs['ksize'], stride, kwargs['padding']),
                                          activation_fns,
                                          module(params[1], params[1], kwargs['ksize'], 1, kwargs['padding']),
                                       )]

            if self.ifgate:
              binary_gate_layers += [nn.Sequential(
                                    module(params[1]+self.embed_dim, params[1], kwargs['ksize'], stride, kwargs['padding']),
                                    activation_fns,
                                    module(params[1], params[1], kwargs['ksize'], 1, kwargs['padding']),
                                    nn.Sigmoid()
                                  )]

            #else:
            #  binary_passing_layers += [nn.Sequential(
            #                              module(params[1] + self.embed_dim, params[1], 5, stride, 2),
            #                              activation_fns,
            #                              module(params[1], params[1], 5, 1, 2),
            #                              activation_fns,
            #                              module(params[1], params[1], 5, 1, 2),
            #                           )]
        self.binary_passing = nn.Sequential(*binary_passing_layers)
        if self.ifgate:
          self.binary_gate = nn.Sequential(*binary_gate_layers)
        self.E = None
        self.graphs = None
        self.masks = None
        self.embed_info = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_embed_info(self, embed_info):
        self.embed_info = embed_info.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()
 
    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        N, K, C, H, W = x.size()
        h = self.conv_map(t, x.view(N*K, C, H, W))
        
        if not self.conv_map.transpose:
          H = H // self.conv_map.stride
          W = W // self.conv_map.stride
        else:
          H = H * self.conv_map.stride
          W = W * self.conv_map.stride
        h = h.view(N, K, -1, H, W)
        h_base = h.clone()
        C = h.size(2)
        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        h_base = h
        # h: [N, K, C, H, W]
        h_u = h.view(N*K, C, H, W)
        if self.embed_flag:
          embed_info = self.embed_info.cuda().expand(N, K, self.embed_info.shape[-1])
          node_info_embed = self.embedding(embed_info).view(N, K, -1, 1, 1).expand(N, K, -1, H, W)
          h_u = torch.cat([h_u, node_info_embed.view(N*K, -1, H, W)], dim=1)

        unary_message = self.unary_passing(h_u.view(N*K, -1, H, W))
        if self.ifgate:
          unary_gate = self.unary_gate(h_u.view(N*K,-1,H,W))
          unary_message = unary_message * unary_gate
        unary_message = unary_message.view(N, K, C, H, W)

        binary_message = torch.zeros(N*K*K, C, H, W).cuda()
        for f_idx in E_matrix.unique():
          f_idx = int(f_idx.item())
          if f_idx == 0:
            continue
          # up or down
          elif f_idx == 1 or f_idx == 2:
            h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C, H, W), \
                                  h.unsqueeze(2).expand(N, K, K, C, H, W)], dim=4) 
            if self.embed_flag:
              node_matrix = torch.cat([node_info_embed.unsqueeze(1).expand(N, K, K, -1, H, W), \
                                       node_info_embed.unsqueeze(2).expand(N, K, K, -1, H, W)], dim=4)
              h_matrix = torch.cat([h_matrix, node_matrix], dim=3)
            h_C = h_matrix.shape[3]
            h_matrix = h_matrix.contiguous().view(N*K*K, h_C, 2*H, W)[E_matrix.contiguous().view(N*K*K, 1, 1, 1).expand(N*K*K, h_C, 2*H, W)==f_idx].contiguous().view(-1, h_C, 2*H, W)
          # left or right
          elif f_idx == 3 or f_idx == 4:
            h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C, H, W), \
                                  h.unsqueeze(2).expand(N, K, K, C, H, W)], dim=5) 
            if self.embed_flag:
              node_matrix = torch.cat([node_info_embed.unsqueeze(1).expand(N, K, K, -1, H, W), \
                                       node_info_embed.unsqueeze(2).expand(N, K, K, -1, H, W)], dim=5)
              h_matrix = torch.cat([h_matrix, node_matrix], dim=3)
            h_C = h_matrix.shape[3]
            h_matrix = h_matrix.contiguous().view(N*K*K, h_C, H, 2*W)[E_matrix.contiguous().view(N*K*K, 1, 1, 1).expand(N*K*K, h_C, H, 2*W)==f_idx].contiguous().view(-1, h_C, H, 2*W)

          binary_message_update = self.binary_passing[f_idx](h_matrix)
          if self.ifgate:
            binary_gate = self.binary_gate[f_idx](h_matrix)
            binary_message_update = binary_message_update * binary_gate
          binary_message[E_matrix.contiguous().view(N*K*K, 1, 1, 1).expand(N*K*K, C, H, W)==f_idx] = binary_message_update.flatten()

        E_pos = (E_matrix > 0).float()
        binary_message = binary_message.view(N, K, K, C, H, W).sum(1) / (1 + E_pos.sum(1).view(N, K, 1, 1, 1).expand(N, K, C, H, W))
        h = unary_message + binary_message

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1, 1, 1).expand(N, K, C, H, W)

        h = h + h_base

        return h

class GraphGatedLayerLinear(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate=True, num_gates=4,num_func=2,reshape=False,embed_config=[False, 0, 0]):
        super(GraphGatedLayerLinear, self).__init__()
        self.linear_map = base_layer(params[0], params[1])
        #kwargs = params[3]
        #module = nn.ConvTranspose2d if kwargs['transpose'] else nn.Conv2d
        module = nn.Linear
        self.num_gates = num_gates
        self.embed_flag  = embed_config[0]
        self.embed_indim = embed_config[1]
        self.embed_dim   = 0
        self.ifgate = ifgate
        self.num_func = num_func

        if self.embed_flag:
          self.embed_dim   = embed_config[2]
          self.embedding = nn.Linear(self.embed_indim, self.embed_dim)

        self.unary_passing   = nn.Sequential(
                                 module(params[1] + self.embed_dim, params[1]),
                                 activation_fns,
                                 module(params[1], params[1])
                               )

        binary_passing_layers = []

        for i in range(1, num_func):
            binary_passing_layers += [nn.Sequential(
                                          module(2*params[1] + 3*self.embed_dim, params[1]),
                                          activation_fns,
                                          module(params[1], params[1])
                                       )]

        if self.ifgate==True:
          self.binary_gate = nn.Sequential(nn.Linear(num_gates,num_gates),nn.Sigmoid())
              
        self.binary_passing = nn.Sequential(*binary_passing_layers)
 
        self.E = None
        self.graphs = None
        self.masks = None
        self.embed_info = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_embed_info(self, embed_info):
        self.embed_info = embed_info.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()
 
    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x) :
        binary_gate = x[:,:,-self.num_gates:]
        x =  x[:,:,:(x.size(2) - self.num_gates)]
        h = self.linear_map(t, x)
        # h: [N, K, C]
        N, K, C = h.size()
        h_base = h.clone()

        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        if self.graphs is not None:
          if self.graphs.shape[0] == 1:
            graphs = self.graphs.cuda().expand(N, K)
          obj_embed = self.embed_obj(graphs)
          unary_message = self.unary_passing(torch.cat([h, obj_embed], dim=-1))

          h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                                h.unsqueeze(2).expand(N, K, K, C), \
                                self.embed_rel(E_matrix.long()), \
                                obj_embed.unsqueeze(1).expand(N, K, K, -1), \
                                obj_embed.unsqueeze(2).expand(N, K, K, -1)], dim=3)
        else:
          unary_message = self.unary_passing(h)
          h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                                h.unsqueeze(2).expand(N, K, K, C)], dim=3)

        binary_message = torch.zeros(N*K*K, C).cuda()
        binary_passing_out = self.binary_passing(h_matrix.view(N*K*K,h_matrix.shape[-1]))
        binary_message = binary_message + (self.binary_passing(h_matrix.view(N*K*K, h_matrix.shape[-1])) * (E_matrix>0).float().view(N*K*K, 1).expand(N*K*K, C))
       
        binary_gate = self.binary_gate(binary_gate.view(N*K,self.num_gates))
        binary_gate = binary_gate.view(N,K,self.num_gates) 
        E_pos = (E_matrix > 0).float()
        binary_message = binary_message.view(N, K, K, C).sum(1) / (1 + E_pos.sum(1).view(N, K, 1).expand(N, K, C))
        for i in range(0,self.num_gates):
          multiplier = binary_gate[:,:,i].unsqueeze(2).expand(N,K,C//self.num_gates)
          binary_message[:,:,i*C//self.num_gates:(i+1)*C//self.num_gates] *= multiplier

        h = unary_message + binary_message
        h = h + h_base
        h_gate = torch.cat((h,binary_gate),dim=2)
        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)
        return h_gate

class GraphLayerLinearGraphGen(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate):
        super(GraphLayerLinearGraphGen, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        
        self.ifgate     = ifgate

        if ifgate:
          assert False, 'gate option is not available here'

        self.unary_passing  = nn.Sequential(
                                nn.Linear(params[1], params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )
        self.binary_passing =  nn.Sequential(
                                   nn.Linear(2*params[1], params[1]),
                                   activation_fns,
                                   nn.Linear(params[1], params[1])
                                )
        self.E = None
        self.graphs = None
        self.masks = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()

    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        h = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        if self.masks is not None:
          mask_matrix = self.masks.unsqueeze(1).expand(N, K, K) * self.masks.unsqueeze(2).expand(N, K, K)
          E_matrix = E_matrix * mask_matrix

        # unary messages
        unary_message = self.unary_passing(h)

        # binary messages
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                              h.unsqueeze(2).expand(N, K, K, C)], dim=3)
        binary_message = torch.zeros(N * K * K, C).cuda()
        binary_message += self.binary_passing(h_matrix.view(N * K * K, 2 * C)) * \
                          (E_matrix > 0).float().view(N * K * K, 1).expand(N * K * K, C)

        # normalize messages
        E_pos = (E_matrix > 0).float()
        binary_message = binary_message.view(N, K, K, C).sum(1) / (1 + E_pos.sum(1).view(N, K, 1).expand(N, K, C))

        # aggregate information
        h = unary_message + binary_message

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)

        return h


class GraphLayerLinearNRI(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate, edge_types):
        super(GraphLayerLinearNRI, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        
        self.ifgate     = ifgate
        self.edge_types = edge_types

        if ifgate:
          assert False, 'gate option is not available here'

        self.unary_passing  = nn.Sequential(
                                nn.Linear(params[1], params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )
        self.binary_passing = []
        for n in range(0, edge_types):
          self.binary_passing += [nn.Sequential(
                                        nn.Linear(2*params[1], params[1]),
                                        activation_fns,
                                        nn.Linear(params[1], params[1])
                                  )]
        self.binary_passing = nn.ModuleList(self.binary_passing)
        self.E = None
        self.graphs = None
        self.masks = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()

    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        h = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        if self.masks is not None:
          mask_matrix = self.masks.unsqueeze(1).expand(N, K, K) * self.masks.unsqueeze(2).expand(N, K, K)
          E_matrix = E_matrix * mask_matrix

        # unary messages
        unary_message = self.unary_passing(h)

        # binary messages
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                              h.unsqueeze(2).expand(N, K, K, C)], dim=3)
        binary_message = torch.zeros(N * K * K, C).cuda()
        for n in range(0, self.edge_types):
          binary_message += self.binary_passing[n](h_matrix.view(N * K * K, 2 * C)) * \
                            (E_matrix[:,:,:,n]).float().view(N * K * K, 1).expand(N * K * K, C)

        # normalize messages
        binary_message = binary_message.view(N, K, K, C).sum(1) / K

        # aggregate information
        h = unary_message + binary_message

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)

        return h


class GraphLayerLinear(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate, num_func=2, reshape=False, embed_config=[0, 0, 0]):
        super(GraphLayerLinear, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        
        self.ifgate     = ifgate
        if ifgate:
          assert False, 'gate option is not available here'

        self.unary_passing  = nn.Sequential(
                                nn.Linear(params[1], params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )
        self.binary_passing =  nn.Sequential(
                                   nn.Linear(2*params[1], params[1]),
                                   activation_fns,
                                   nn.Linear(params[1], params[1])
                                )
        self.E = None
        self.graphs = None
        self.masks = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()

    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        h = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        if self.masks is not None:
          mask_matrix = self.masks.unsqueeze(1).expand(N, K, K) * self.masks.unsqueeze(2).expand(N, K, K)
          E_matrix = E_matrix * mask_matrix

        # unary messages
        unary_message = self.unary_passing(h)

        # binary messages
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                              h.unsqueeze(2).expand(N, K, K, C)], dim=3)
        binary_message = torch.zeros(N * K * K, C).cuda()
        binary_message += self.binary_passing(h_matrix.view(N * K * K, h_matrix.shape[-1])) * \
                          (E_matrix > 0).float().view(N * K * K, 1).expand(N * K * K, C)

        # normalize messages
        E_pos = (E_matrix > 0).float()
        binary_message = binary_message.view(N, K, K, C).sum(1) # / (1 + E_pos.sum(1).view(N, K, 1).expand(N, K, C))

        # aggregate information
        h = unary_message + binary_message

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)

        return h


class GraphLayerTransformerLinear(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate, num_func=2, reshape=False, embed_config=[0, 0, 0], num_heads=8):
        super(GraphLayerTransformerLinear, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        
        self.ifgate     = ifgate
        if ifgate:
          assert False, 'gate option is not available here'

        self.unary_transformer  = MultiHeadAttention(num_heads, params[1], params[1], params[1])
        self.binary_transformer = MultiHeadAttention(num_heads, params[1], params[1], params[1])
        self.unary_mlp  = MLP(params[1], 2048, params[1], num_layers=3)
        self.binary_mlp = MLP(params[1], 2048, params[1], num_layers=3)

        self.E = None
        self.graphs = None
        self.masks = None

    @staticmethod
    def _get_fc(N, num_node):
        E_pos = 1 - torch.eye(num_node)
        E_pos = E_pos.unsqueeze(0).expand(N, num_node, num_node).cuda()
        E = E_pos
        return E
 
    def set_E(self, E, shape=None):
        if E is None:
          N, num_node = shape
          self.E = self._get_fc(N, num_node)
        else:
          self.E = E.cpu()

    def set_graphs(self, graphs):
        self.graphs = graphs.cpu()

    def set_masks(self, masks):
        self.masks = masks

    def forward(self, t, x):
        h = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        if self.E is None:
          E_matrix = self._get_fc(N, K)
        elif self.E.shape[0] == 1:
          E_matrix = self.E.cuda().expand(N, K, K)
        else:
          E_matrix = self.E.cuda()

        if self.masks is not None:
          mask_matrix = self.masks.unsqueeze(1).expand(N, K, K) * self.masks.unsqueeze(2).expand(N, K, K)
          E_matrix = E_matrix * mask_matrix

        unary, attns  = self.unary_transformer(h, h, h, mask=self.masks, mask_matrix=mask_matrix)
        binary, attns = self.binary_transformer(h, h, h, mask=self.masks, mask_matrix=mask_matrix)
        unary         = self.unary_mlp(unary)
        binary        = self.unary_mlp(binary)
        output        = unary + binary

        if self.masks is not None:
          h = h * self.masks.view(N, K, 1).expand(N, K, C)

        return h


# define MutiHeadAttention
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.0))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, mask_matrix=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        rep_mask_matrix = mask_matrix.repeat(n_head, 1, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask_matrix=rep_mask_matrix)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        if mask is not None:
          output = output * mask.unsqueeze(-1).expand(output.shape)
        return output, attn



# define MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(MLP, self).__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_size  = hidden_size
        self.layers = [nn.Linear(input_size, hidden_size)]
        for i in range(0, num_layers):
          #self.layers += [nn.ReLU()]
          self.layers += [nn.Softplus()]
          self.layers += [nn.Linear(hidden_size, hidden_size)]
        self.layers += [nn.Linear(hidden_size, output_size)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.model(x)
        return output


# module for Transformer
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask_matrix=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask_matrix is not None:
            mask_matrix = mask_matrix.view(attn.shape)
            attn = attn.masked_fill(mask_matrix==0, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        if mask_matrix is not None:
            attn[mask_matrix==0] = 0
        output = torch.bmm(attn, v)

        return output, attn


class GraphLayerConvSimple(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate):
        super(GraphLayerConvSimple, self).__init__()
        kwargs = params[2]
        module = nn.ConvTranspose2d if kwargs['transpose'] else nn.Conv2d
        self.conv_map = base_layer(params[0], params[1], **kwargs) 
        self.ifgate = ifgate
        if ifgate:
            self.unary_gate     = nn.Sequential(
                                    module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    activation_fns,
                                    module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    nn.Sigmoid()
                                  )
            self.binary_gate    = nn.Sequential(
                                    module(2*params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    activation_fns,
                                    module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                    nn.Sigmoid()
                                  )
        self.unary_passing  = nn.Sequential(
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                activation_fns,
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding'])
                              )
        self.binary_passing = nn.Sequential(
                                module(2*params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding']),
                                activation_fns,
                                module(params[1], params[1], kwargs['ksize'], kwargs['stride'], kwargs['padding'])
                              )

    def forward(self, t, x):
        N, K, C, H, W = x.size()
        h = self.conv_map(t, x.view(N*K, C, H, W)).view(N, K, -1, H, W)
        C = h.size(2)

        if self.E is None:
          E_matrix = 1 - torch.eye(K).view(1, K, K).cuda()
          E_matrix = E_matrix.expand(N, K, K)
        else:
          E_matrix = self.E

        # h: [N, K, C, H, W]
        unary_message = self.unary_passing(h.view(N*K, C, H, W))
        if self.ifgate:
          unary_gate    = self.unary_gate(h.view(N*K, C, H, W))
          unary_message = unary_message * unary_gate
        unary_message = unary_message.view(N, K, C, H, W)
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C, H, W), \
                              h.unsqueeze(2).expand(N, K, K, C, H, W)], dim=3)
        binary_message = self.binary_passing(h_matrix.view(N*K*K, 2*C, H, W))
        if self.ifgate:
          binary_gate    = self.binary_gate(h_matrix.view(N*K*K, 2*C, H, W))
          binary_message = binary_message * binary_gate
        binary_message = binary_message.view(N, K, K, C, H, W).sum(1) / E_matrix.sum(1).view(N, K, 1, 1, 1).expand(N, K, C, H, W)
        h = unary_message + binary_message

        return h


class GraphLayerLinearSimple(nn.Module):
    def __init__(self, base_layer, params, activation_fns, ifgate):
        super(GraphLayerLinearSimple, self).__init__()
        self.linear_map = base_layer(params[0], params[1]) 
        self.ifgate     = ifgate
        if ifgate:
            self.unary_gate  = nn.Sequential(
                                    nn.Linear(params[1], params[1]),
                                    activation_fns,
                                    nn.Linear(params[1], params[1]),
                                    nn.Sigmoid()
                                  )
            self.binary_gate = nn.Sequential(
                                    nn.Linear(2*params[1], params[1]),
                                    activation_fns,
                                    nn.Linear(params[1], params[1]),
                                    nn.Sigmoid()
                                  )
        self.unary_passing  = nn.Sequential(
                                nn.Linear(params[1], params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )
        self.binary_passing = nn.Sequential(
                                nn.Linear(2*params[1], params[1]),
                                activation_fns,
                                nn.Linear(params[1], params[1])
                              )

    def forward(self, t, x):
        h   = self.linear_map(t, x)

        # h: [N, K, C]
        N, K, C = h.size()

        E_matrix = 1 - torch.eye(K).view(1, K, K).cuda()
        E_matrix = E_matrix.expand(N, K, K)

        unary_message = self.unary_passing(h)
        if self.ifgate:
          unary_gate    = self.unary_gate(h)
          #unary_gate    = torch.softmax(unary_gate, dim=-1)
          unary_message = unary_message * unary_gate
        h_matrix = torch.cat([h.unsqueeze(1).expand(N, K, K, C), \
                              h.unsqueeze(2).expand(N, K, K, C)], dim=3)
        binary_message = self.binary_passing(h_matrix.view(N*K*K, 2*C))
        if self.ifgate:
          binary_gate    = self.binary_gate(h_matrix.view(N*K*K, 2*C))
          #binary_gate    = torch.softmax(binary_gate, dim=-1)
          binary_message = binary_message * binary_gate
        binary_message = binary_message.view(N, K, K, C).sum(1) / E_matrix.sum(1).view(N, K, 1).expand(N, K, C)
        h = unary_message + binary_message

        return h



