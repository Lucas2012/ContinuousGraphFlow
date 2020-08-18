import torch.nn as nn
from torch.distributions import *
import torch
from . import diffeq_layers
from .cnf import *
from .odefunc import *
import torch.nn.functional as F


def inverse_sigmoid(x):
    return -torch.log(torch.reciprocal(x) - 1.)

def sumflat(x):
    return torch.sum(x, dim=-1, keepdim=True)

class Sigmoid():
    def forward(self, x):
        y = F.sigmoid(x)
        logd = -F.softplus(x) - F.softplus(-x)
        return y, sumflat(logd)
    def inverse(self, y):
        x = inverse_sigmoid(y)
        logd = -torch.log(y) - torch.log(1. - y)
        return x, sumflat(logd)


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList, prior_config=[False, 0, 0], variational_dequantization=False, onehot=True):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)
        self.require_label = prior_config[0]
        self.num_label     = prior_config[1]
        self.embed_dim     = prior_config[2]
        self.vd            = variational_dequantization
        if self.require_label:
            if onehot:
              self.mean_emb    = nn.Embedding(self.num_label, self.embed_dim)
              self.log_var_emb = nn.Embedding(self.num_label, self.embed_dim)
              self.mean_emb.weight.data.zero_()
              self.log_var_emb.weight.data.zero_()
            else:
              self.mean_emb    = nn.Linear(self.num_label, self.embed_dim)
              self.log_var_emb = nn.Linear(self.num_label, self.embed_dim)
              self.mean_emb.weight.data.zero_()
              self.log_var_emb.weight.data.zero_()

        if self.vd:
            self.dequantizer_mu     = nn.Parameter(torch.zeros(size=(1,)))
            self.dequantizer_logvar = nn.Parameter(torch.zeros(size=(1,)))
            self.sigmoid            = Sigmoid()

    def set_embed_info(self, embed_info):
        for l in self.chain:
          if isinstance(l, CNF):
            if isinstance(l.odefunc.odefunc.diffeq, ODEGraphnet):
              l.odefunc.odefunc.diffeq.set_embed_info(embed_info)

    def set_graphs(self, graphs):
        for l in self.chain:
          if isinstance(l, CNF):
            if isinstance(l.odefunc.odefunc.diffeq, ODEGraphnet)  \
               or isinstance(l.odefunc.odefunc.diffeq, ODEGraphnetGraphGen):
                l.odefunc.odefunc.diffeq.set_graphs(graphs)

    def set_E(self, E, shape=None):
        for l in self.chain:
          if isinstance(l, CNF):
            if isinstance(l.odefunc.odefunc.diffeq, ODEGraphnet) \
               or isinstance(l.odefunc.odefunc.diffeq, ODEGraphnetGraphGen):
                l.odefunc.odefunc.diffeq.set_E(E, shape)

    def set_masks(self, masks):
        for l in self.chain:
          if isinstance(l, CNF):
            if isinstance(l.odefunc.odefunc.diffeq, ODEGraphnet) \
               or isinstance(l.odefunc.odefunc.diffeq, ODEGraphnetGraphGen):
              l.odefunc.odefunc.diffeq.set_masks(masks)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx

    def get_logp(self, z, label):
        # label: [N, 1]
        # z:     [N, D]

        mean    = self.mean_emb(label)
        log_var = self.log_var_emb(label)

        priors  = Normal(mean, torch.exp(log_var))
        return priors.log_prob(z)

    def get_prior_samples(self, label):
        # label: [N, K, 1]

        mean    = self.mean_emb(label)
        log_var = self.log_var_emb(label)

        priors  = Normal(mean, torch.exp(log_var))
        return priors.sample()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        esp = torch.randn(*mu.size()).cuda()
        dist = Normal(mu[0,0], torch.exp(logvar)[0,0])
        logp_esp = dist.log_prob(esp.view(-1, 1)).view(mu.shape)
        z = mu + std * esp

        p_z = logp_esp - std.sum(-1, keepdim=True)

        return z, p_z

    def dequantizer(self, x, K):
        mu, logvar       = self.dequantizer_mu.view(1,1,1).expand(x.shape), self.dequantizer_logvar.view(1,1,1).expand(x.shape)
        unboundu, logpu  = self.reparameterize(mu, logvar)
        u, sigmoid_logpu = self.sigmoid.forward(unboundu)

        x = (x + u) * (K - 1) / K

        return x, (logpu - sigmoid_logpu).sum(-1).sum(-1, keepdim=True)
