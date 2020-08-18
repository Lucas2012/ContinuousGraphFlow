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

import lib.ffjord.layers as layers
import lib.ffjord.utils as utils
import lib.ffjord.odenvp as odenvp
import lib.ffjord.multiscale_parallel as multiscale_parallel

from lib.ffjord.train_misc import standard_normal_logprob
from lib.ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from lib.ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from lib.ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

from lib.utils import get_grid_configuration
from lib.utils import int_tuple, bool_flag, str_tuple 
from lib.utils import generate_layout_samples
from lib.utils import generate_conditional_layout_samples

from lib.data_loaders.vg import *
from lib.data_loaders.coco import *

torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser = argparse.ArgumentParser("Continuous Graph Flow")
parser.add_argument("--data", choices=["vg", "coco"], type=str, default="coco")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=3, help='Number of stacked CNFs.')
parser.add_argument("--num_func", type=int, default=-1, help='number of functions.')
parser.add_argument("--embed_dim", type=int, default=64, help='embedding dimensions.')
parser.add_argument("--require_label", type=eval, default=True, choices=[True, False], help='Whether to use label to drive learnable prior')
parser.add_argument("--if_graph_gate", type=eval, default=False, choices=[True, False], help='Whether to use gate in graph')
parser.add_argument("--if_graph_variable", type=eval, default=True, choices=[True, False], help='Whether to use variable graphs')
parser.add_argument("--use_semantic_graph", type=eval, default=False, choices=[True, False], help='Whether to use semantic graph edge matrix')


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

parser.add_argument("--imagesize", default='64,64', type=int_tuple)
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
)
parser.add_argument("--test_batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)

parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--graphflow', type=eval, default=True, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])
parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
parser.add_argument('--sample_mode', type=eval, default=False, choices=[True, False])

# Regularizations
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
parser.add_argument(
    "--max_grad_norm", type=float, default=100,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/video_graph/graph_flow")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)

# visual genome dataset arguments
VG_DIR = os.path.expanduser('data/vg')

# common to coco and VG
parser.add_argument('--num_val_samples', default=1024, type=int)

# VG-specific options
parser.add_argument('--include_relationships', default=True, type=bool_flag)
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)
parser.add_argument('--num_train_samples', default=None, type=int)


# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default='data/coco/images/train2017')
parser.add_argument('--coco_val_image_dir',
         default='data/coco/images/val2017')
parser.add_argument('--coco_train_instances_json',
         default='data/coco/annotations/instances_train2017.json')
parser.add_argument('--coco_train_stuff_json',
         default='data/coco/annotations/stuff_train2017.json')
parser.add_argument('--coco_val_instances_json',
         default='data/coco/annotations/instances_val2017.json')
parser.add_argument('--coco_val_stuff_json',
         default='data/coco/annotations/stuff_val2017.json')
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)


args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


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
    if args.data == 'vg':
        collate_fn = vg_collate_fn
    elif args.data == 'coco':
        collate_fn = coco_collate_fn
    train_loader = build_loaders(train_set, args.batch_size, 4, True, True, True, collate_fn)
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def get_dataset(args):
    if args.data == "vg":
        vocab, train_set, test_set = build_vg_dsets(args)
        collate_fn = vg_collate_fn
    elif args.data == "coco":
        vocab, train_set, test_set = build_coco_dsets(args)
        collate_fn = coco_collate_fn
    else:
        raise ValueError('Not implemented yet.')

    data_shape = (4,)

    test_loader = build_loaders(train_set, args.batch_size, 4, True, True, True, collate_fn)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn
    )
    return vocab, train_set, test_loader, data_shape


def compute_bits_per_dim(graphs, graph_boxes, E_matrix, graph_masks, model, args):
    zero = torch.zeros(graph_boxes.shape[0], 1).to(graph_boxes)

    if args.use_semantic_graph:
      model.module.set_graphs(graphs.long() + 1)
      model.module.set_E(E_matrix.long() + 1)
      model.module.set_masks(graph_masks)
    else:
      assert False, 'semantic graph is required for vg scene layout generation'
    z, delta_logp = model(graph_boxes, zero)  # run model forward

    if args.require_label:
      logpz = model.module.get_logp(z, graphs.long() + 1) # logp(z)
      logpz = logpz.sum(-1) * graph_masks
      logpz = logpz.sum(1, keepdim=True)
    else:
      logpz = standard_normal_logprob(z).sum(-1) * graph_masks
      logpz = logpz.sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = - torch.sum(logpx) / (graph_masks >= 0).float().sum().item()  # averaged for per node

    return logpx_per_dim


def create_model(args, data_shape, regularization_fns, vocab_size):
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
            def build_cnf():
                graphode_diffeq = layers.ODEGraphnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=False,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                    ifgate=args.if_graph_gate,
                    num_func=args.num_func,
                    embed_config=[args.embed_dim>0, args.num_func, vocab_size, args.embed_dim]
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
            print('************** Not using graph flow!!! ****************')
            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=False,
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

        chain = []
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        data_nelements = 1
        for ele in data_shape:
          data_nelements *= ele
        model = layers.SequentialFlow(chain, prior_config=[args.require_label, \
                                                           vocab_size, \
                                                           data_nelements])
    return model

if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    vocab, train_set, test_loader, data_shape = get_dataset(args)

    args.num_func = len(vocab['pred_name_to_idx']) + 1
    print('set number of functions to be pred_types + 1: ', args.num_func)
    vocab_obj_size = 0
    for k in vocab['object_name_to_idx'].keys():
      v = vocab['object_name_to_idx'][k]
      if v > vocab_obj_size:
        vocab_obj_size = v
    if args.data == 'vg':
      vocab_obj_size += 1
    print('set number of objects to be obj_types + 1: ', vocab_obj_size + 1)
    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns, vocab_obj_size + 1)

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

    if args.require_label:
        # For visualization.
        test_iter = iter(test_loader)
        fixed_graphs, fixed_graph_matrix, _, fixed_graph_masks, fixed_objs, fixed_triples, fixed_obj_to_img, fixed_triple_to_img = next(test_iter)
        fixed_z        = cvt(model.module.get_prior_samples(cvt(fixed_graphs + 1).long()))
        fixed_E_matrix = cvt(fixed_graph_matrix)
        fixed_masks    = cvt(fixed_graph_masks)
        fixed_graphs   = cvt(fixed_graphs)
    else:
        assert False

    if args.sample_mode:
        generate_layout_samples(model=model, loader=test_loader, num_samples=400, output_dir=args.data + '_generation_results', vocab=vocab, num_vis=100)
        generate_conditional_layout_samples(model=model, loader=test_loader, num_samples=400, output_dir=args.data + '_conditional_generation_results', vocab=vocab, num_vis=50)

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    best_loss = float("inf")
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set, epoch)
        for _, (graphs, graph_matrix, graph_boxes, graph_masks, _, _, _, _) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()
        
            # cast data and move to device
            graphs, graph_matrix, graph_boxes, graph_masks = [cvt(x) for x in [graphs, graph_matrix, graph_boxes, graph_masks]]
            # compute loss
            loss = compute_bits_per_dim(graphs, graph_boxes, graph_matrix, graph_masks, model, args)
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
                itr = 0
                for (graphs, graph_matrix, graph_boxes, graph_masks, _, _, _, _) in test_loader:
                    graphs, graph_matrix, graph_boxes, graph_masks = [cvt(x) for x in [graphs, graph_matrix, graph_boxes, graph_masks]]
                    loss = compute_bits_per_dim(graphs, graph_boxes, graph_matrix, graph_masks, model, args)
                    losses.append(loss)
                    itr += 1

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

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            model.module.set_graphs(fixed_graphs.long() + 1)
            model.module.set_E(fixed_E_matrix.long() + 1)
            model.module.set_masks(fixed_masks)
            generated_samples = model(fixed_z, reverse=True)
            # visualize the bounding boxes
            print('To be filled: visualizing generated bounding boxes.')
