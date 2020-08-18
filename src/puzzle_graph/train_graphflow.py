import argparse
import os
import time
import numpy as np
import random

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image
from torch.distributions import Normal

import lib.ffjord.layers as layers
import lib.ffjord.utils as utils
import lib.ffjord.odenvp as odenvp
import lib.ffjord.graph_odenvp as graph_odenvp
import lib.ffjord.multiscale_parallel as multiscale_parallel

from lib.ffjord.train_misc import standard_normal_logprob
from lib.ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from lib.ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from lib.ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

from lib.utils import get_grid_configuration
from lib.utils import get_embed_info

from lib.data_loaders.celeba_hq import CelebaHQDataset as celeba

torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser = argparse.ArgumentParser("Continuous Graph Flow")
parser.add_argument("--data", choices=["cityscapes_puzzle", "fashionmnist_puzzle", "mnist_video", "mnist_puzzle", "coco_puzzle", "celeba_puzzle","multimnist_puzzle","cifar_puzzle","lsun_puzzle"], type=str, default="mnist_puzzle")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--aug_dim", type=int, default="0")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument("--puzzle_size", type=int, default=2, help='size of puzzle.')
parser.add_argument("--embed_dim", type=int, default=64, help='embed dimension for node label.')
parser.add_argument("--patch_size", type=int, default=16, help='size of patch.')
parser.add_argument("--num_layers", type=int, default=3, help='number of functions.')
parser.add_argument("--require_label", type=eval, default=True, choices=[True, False], help='Whether to use label to drive learnable prior')
parser.add_argument("--if_graph_gate", type=eval, default=False, choices=[True, False], help='Whether to use gates in graph layers')
parser.add_argument("--if_graph_variable", type=eval, default=False, choices=[True, False], help='Whether to use variable graphs')
parser.add_argument("--use_semantic_graph", type=eval, default=False, choices=[True, False], help='Whether to use semantic graph edge matrix')


parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument("--conv_embed", type=eval, default=False, choices=[True, False])
parser.add_argument(
    "--layer_type", type=str, default="ignore",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
)
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--normalize", type=eval, default=False, choices=[True, False])
parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--graphflow', type=eval, default=False, choices=[True, False])
parser.add_argument('--ifpuzzle', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])
parser.add_argument('--graph_multiscale', type=eval, default=False, choices=[True, False])
parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
parser.add_argument('--permute', type=eval, default=False, choices=[True, False])
parser.add_argument('--node_autoencode', type=eval, default=False, choices=[True, False])
#parser.add_argument('--graphflow_autoencode', type=eval, default=False, choices=[True, False])
#parser.add_argument('--autoencodesimple', type=eval, default=False, choices=[True, False])
parser.add_argument('--method', type=str, default="", choices=["plain", "autoencodesimple", "gated","graphflow_autoencode"])
parser.add_argument('--multiscale_method', type=str, default="", choices=["linear", "conv", "ae"])
parser.add_argument('--mp_type',type=str,default="generic",choices=["affine","generic","augmented"])

# Regularizations
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
parser.add_argument(
    "--max_grad_norm", type=float, default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/video_graph/graph_flow")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)

parser.add_argument('--train', type=eval, default=True, choices=[True, False])
parser.add_argument('--visualize', type=eval, default=False, choices=[True, False])
parser.add_argument('--temp', type=float, default=0.7)
parser.add_argument('--n_samples', type=int, default=30)
parser.add_argument('--num_func',type=int,default=5)
parser.add_argument('--num_gates',type=int,default=4)


args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize((im_size,im_size)), tforms.ToTensor(), add_noise])

    if args.data == "mnist_puzzle":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        if args.imagesize is None:
          args.imagesize = 28
        train_set = dset.MNIST(root="./data/mnist/MNIST/", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data/mnist/MNIST/", train=False, transform=trans(im_size), download=True)
    elif args.data == "fashionmnist_puzzle":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        if args.imagesize is None:
          args.imagesize = 28
        train_set = dset.FashionMNIST(root="./data/fashionmnist/fashionMNIST/", train=True, transform=trans(im_size), download=True)
        test_set = dset.FashionMNIST(root="./data/fashionmnist/fashionMNIST/", train=False, transform=trans(im_size), download=True)
    elif args.data == "coco_puzzle":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        if args.imagesize is None:
          args.imagesize = 64
        train_set = dset.CocoDetection("./data/coco/images/train2017/","./data/coco/annotations/annotations/instances_train2017.json", transform=trans(im_size)) 
        test_set = dset.CocoDetection("./data/coco/images/val2017/","./data/coco/annotations/annotations/instances_val2017.json" , transform=trans(im_size))
    elif args.data == "celeba_puzzle":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = celeba('./data/celeba_hq/imgs/','./data/celeba_hq/splits/','train',im_size,normalize=args.normalize,noise=args.add_noise)
        test_set = celeba('./data/celeba_hq/imgs/','./data/celeba_hq/splits/','test',im_size,normalize=args.normalize,noise=args.add_noise)
    elif args.data == "cityscapes_puzzle":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.Cityscapes(root="./data/cityscapes/train", \
                                    split='train', \
                                    mode='fine', \
                                    target_type='instance', \
                                    transform=trans(im_size), \
                                    target_transform=None)
        test_set  = dset.Cityscapes(root="./data/cityscapes/test", \
                                    split='test', \
                                    mode='fine', \
                                    target_type='instance', \
                                    transform=trans(im_size), \
                                    target_transform=None)

    elif args.data == "cifar_puzzle":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        if args.imagesize is None:
          args.imagesize = 64
        train_set = dset.CIFAR10(root="./data/cifar10/", train=True, transform=trans(im_size), download=True)
        test_set = dset.CIFAR10(root="./data/cifar10/", train=False, transform=trans(im_size), download=True)

    elif args.data == "lsun_puzzle":
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        if args.imagesize is None:
          args.imagesize = 64
        train_set = dset.LSUN('./data/lsun/',['bedroom_train'],transform=trans(im_size))
        test_set = dset.LSUN('./data/lsun/',['bedroom_val'],transform=trans(im_size))

    data_shape = (im_dim, im_size//args.puzzle_size, im_size//args.puzzle_size)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model, args, E_matrix=None):
    if args.ifpuzzle:
        x_original = x.clone()
        puzzle_size = args.puzzle_size

        assert (args.imagesize == args.puzzle_size * args.patch_size),"Check image size and puzzle size"
        puzzle_H = args.patch_size
        puzzle_W = args.patch_size
 
        N, C, H, W = x.size()
        puzzle = []

        label = torch.arange(0, puzzle_size * puzzle_size).view(1, puzzle_size, puzzle_size)
        label = label.expand(x.size(0), puzzle_size, puzzle_size).long().cuda()

        if args.if_graph_variable is True:
          start_h = random.randint(0, puzzle_size-2)
          start_w = random.randint(0, puzzle_size-2)
          end_h   = random.randint(start_h+1, puzzle_size)
          end_w   = random.randint(start_w+1, puzzle_size)
          label   = label[:, start_h:end_h, start_w:end_w]
        else:
          start_h, start_w, end_h, end_w = 0, 0, puzzle_size, puzzle_size
        for hi in range(start_h, end_h):
          for wi in range(start_w, end_w):
            puzzle += [x[:,:,hi*puzzle_H:(hi+1)*puzzle_H,wi*puzzle_W:(wi+1)*puzzle_W].unsqueeze(1)]
        label = label.contiguous().view(x.size(0), -1)

        x = torch.cat(puzzle, dim=1)

        if args.permute:
          for bi in range(0, N):
            perm_bi = torch.randperm((end_h - start_h)*(end_w - start_w))
            x[bi]   = x[bi,perm_bi]
            label[bi] = label[bi,perm_bi]

    
    if args.graphflow is False:
      x = x.view([x.shape[0]*x.shape[1]] + list(x.shape)[2:])
    zero = torch.zeros(x.shape[0], 1).to(x)
    if args.if_graph_gate == True:
      gb = torch.ones(x.shape[0],x.shape[1],args.num_gates) * (1.0/(args.num_gates))
      gb = gb.cuda() 

    if args.use_semantic_graph == True:
      model.module.set_graphs(label[0:1,...])
      model.module.set_E(get_grid_configuration(label, args.puzzle_size)[0:1,...])
      model.module.set_embed_info(get_embed_info(label, args.puzzle_size)[0:1,...])

    if args.if_graph_gate == False:
      z, delta_logp = model(x, zero)  # run model forward
    elif args.if_graph_gate == True and args.method=="gated":
      N,K,C,H,W = x.shape
      x_in = x.view(N,K,C*H*W)
      z, delta_logp = model(torch.cat((x_in,gb),dim=2),zero)
    
    if args.graphflow is False:
      assert False, 'Just warning that graphflow is not used.'
      logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    else:
      if args.require_label:
        if isinstance(z, list):
          logpz = model.module.get_logp(z, label.view(x.shape[0], x.shape[1])) # logp(z)
        else:
          if args.method == "gated":
            logpz = model.module.get_logp(z.view(z.shape[0], z.shape[1], -1)[:,:,:z.shape[2]-args.num_gates], label.view(z.shape[0], z.shape[1])) # logp(z)
          else:
            logpz = model.module.get_logp(z.view(z.shape[0], z.shape[1], -1), label.view(z.shape[0], z.shape[1])) # logp(z)
          logpz = logpz.sum(1).sum(1, keepdim=True)
      else:
        logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        assert False, "Not graph flow -- in case running the wrong one."
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
        )
    elif args.graph_multiscale:
       if (args.multiscale_method == "conv" or args.multiscale_method == "linear") and args.mp_type == "generic" and args.if_graph_gate == False:
         network_choice = layers.ODEGraphnet
       elif (args.multiscale_method == "conv" or args.multiscale_method == "linear") and args.mp_type == "generic" and args.if_graph_gate==True:
         network_choice = layers.ODEGatedGraphnet

       prior_config = [args.require_label, args.puzzle_size**2]
       conv_embed_config = [args.conv_embed, 2, 32]
       model = graph_odenvp.GraphODENVP(
            (args.batch_size,0, *data_shape),
            strides= strides,
            n_blocks=args.num_blocks,
            threshold_output_size=args.patch_size//4,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            unit_type = args.multiscale_method,
            mp_type = args.mp_type,
            conv=args.conv,
            ifgate=args.if_graph_gate,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
            network_choice=network_choice,
            prior_config=prior_config,
            conv_embed_config=conv_embed_config,
        )
    elif args.parallel:
        assert False, "Not graph flow -- in case running the wrong one."
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.graphflow:
          if args.method == 'plain':
              def build_cnf():
                  graphode_diffeq = layers.ODEGraphnet(
                      hidden_dims=hidden_dims,
                      input_shape=data_shape,
                      strides=strides,
                      conv=args.conv,
                      layer_type=args.layer_type,
                      nonlinearity=args.nonlinearity,
                      ifgate=args.if_graph_gate,
                      node_autoencode=args.node_autoencode,
                      num_func=args.num_func,
                      num_layers=args.num_layers
                  )
                  odefunc = layers.ODEfunc(
                      diffeq=graphode_diffeq,
                      divergence_fn=args.divergence_fn,
                      residual=args.residual,
                      rademacher=args.rademacher,
                  )
                  cnf = layers.CNF(
                      odefunc=odefunc,
                      T=args.time_length,
                      regularization_fns=regularization_fns,
                      solver=args.solver,
                  )
                  return cnf

          elif args.method == 'gated':
              def build_cnf():
                  graphode_diffeq = layers.ODEGatedGraphnet(
                      hidden_dims=hidden_dims,
                      input_shape=data_shape,
                      strides=strides,
                      conv=args.conv,
                      layer_type=args.layer_type,
                      nonlinearity=args.nonlinearity,
                      ifgate=args.if_graph_gate,
                      node_autoencode=args.node_autoencode,
                      num_func=args.num_func,
                      num_layers=args.num_layers
                  )
                  odefunc = layers.ODEfunc(
                      diffeq=graphode_diffeq,
                      divergence_fn=args.divergence_fn,
                      residual=args.residual,
                      rademacher=args.rademacher,
                  )
                  cnf = layers.CNF(
                      odefunc=odefunc,
                      T=args.time_length,
                      regularization_fns=regularization_fns,
                      solver=args.solver,
                  )
                  return cnf
        else:
            assert False
            print('************** Not using graph flow!!! ****************')
            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        data_nelements = 1
        for ele in data_shape:
          data_nelements *= ele
        model = layers.SequentialFlow(chain, prior_config=[args.require_label, \
                                                           args.puzzle_size ** 2, \
                                                           data_nelements])
    return model


if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def cvt(x):
      if isinstance(x, list):
        return [ele.type(torch.float32).to(device, non_blocking=True) for ele in x]
      else:
        return x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_label = cvt(torch.arange(0, args.puzzle_size**2)).unsqueeze(0).expand(10, -1).long()
    fixed_z     = cvt(model.module.get_prior_samples(cvt(fixed_label).long()))
    if not isinstance(fixed_z, list):
      fixed_z = fixed_z.view((-1,args.puzzle_size**2) + data_shape)

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    best_loss = float("inf")
    itr = 0

    if args.train == True:
      for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set, epoch)
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()
        
            # cast data and move to device
            x = cvt(x)
            # compute loss
            loss = compute_bits_per_dim(x, model, args)
            if regularization_coeffs:
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            total_time = count_total_time(model)
            loss = loss + total_time * args.time_penalty
        
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
            optimizer.step()
        
            if args.spectral_norm: spectral_norm_power_iteration(model, args.spectral_norm_niter)
        
            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)
        
            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f}) | lr {:.6f}".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                        steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg, optimizer.param_groups[0]['lr']
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)
        
            itr += 1

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for (x, y) in test_loader:
                    x = cvt(x)
                    loss = compute_bits_per_dim(x, model, args)
                    losses.append(loss)

                losses = [ele.item() for ele in losses]
                loss = np.mean(losses)
                logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))
                utils.makedirs(args.save)
                torch.save({
                    "args": args,
                    "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                }, os.path.join(args.save, "checkpt" + str(epoch) +".pth"))

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            if args.use_semantic_graph == True:
              model.module.set_graphs(fixed_label[0:1,...])
              model.module.set_E(get_grid_configuration(fixed_label, args.puzzle_size)[0:1,...])
              model.module.set_embed_info(get_embed_info(fixed_label, args.puzzle_size)[0:1,...])
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=args.puzzle_size * args.puzzle_size)

    if args.visualize == True:
        for i in range(0,args.n_samples):
          fixed_label = cvt(torch.arange(0, args.puzzle_size**2)).unsqueeze(0).expand(10, -1).long()
          fixed_z     = cvt(model.module.get_prior_samples(cvt(fixed_label).long()))
          if not isinstance(fixed_z, list):
            fixed_z = fixed_z.view((-1,args.puzzle_size**2) + data_shape)
          with torch.no_grad():
            if args.use_semantic_graph == True:
              model.module.set_graphs(fixed_label[0:1,...])
              model.module.set_E(get_grid_configuration(fixed_label, args.puzzle_size)[0:1,...])
              model.module.set_embed_info(get_embed_info(fixed_label, args.puzzle_size)[0:1,...])
            generated_samples = model(fixed_z, reverse=True,temperature=args.temp).view(-1, *data_shape)
            for img_idx in range(0,generated_samples.shape[0]//(args.puzzle_size **2)):
              fig_filename = os.path.join(args.save, "samples"+str(args.temp),"{:04d}.jpg".format((i *10)+1+img_idx))
              utils.makedirs(os.path.dirname(fig_filename))
              img = generated_samples[(img_idx * (args.puzzle_size ** 2)):((img_idx+1) * (args.puzzle_size **2)),:,:,:]
              save_image(img, fig_filename, nrow=args.puzzle_size)
              print("Saved sample",img_idx,"to file",fig_filename)
              for p_i in range(0,img.size(0)):
                patch_filename = os.path.join(args.save, "samples"+str(args.temp), 'patches',str((i*10)+1+img_idx),"{:04d}.jpg".format(p_i))
                utils.makedirs(os.path.dirname(patch_filename))
                patch = img[p_i,:,:,:]
                save_image(patch,patch_filename,nrow=1)
                print("Saving patch",p_i,"to file",patch_filename)
