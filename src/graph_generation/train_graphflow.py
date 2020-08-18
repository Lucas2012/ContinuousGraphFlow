import argparse
import os
import time
import numpy as np
import random

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms

import lib.ffjord.layers as layers
import lib.ffjord.utils as utils

from lib.ffjord.train_misc import standard_normal_logprob
from lib.ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from lib.ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from lib.ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

from lib.utils import get_grid_configuration
from lib.utils import int_tuple, bool_flag, str_tuple 

from lib.data_loaders.graph_generation import *
import create_graphs

from src.graph_generation.utils import *

torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser = argparse.ArgumentParser("Continuous Graph Flow")
parser.add_argument("--dims", type=str, default="64,64")
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument("--model_name", choices=["cgf"], type=str, default="cgf")
parser.add_argument("--note", type=str, default="none")
parser.add_argument("--require_prior", type=eval, default=False, choices=[True, False], help='Whether to define prior distribution in flow model')
parser.add_argument("--use_logit", type=eval, default=False, choices=[True, False], help='Whether to use LogitTransform')


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

parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--num_epochs", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--vd", type=eval, default=False, choices=[True, False])
parser.add_argument(
    "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
)
parser.add_argument("--test_batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_iters", type=float, default=100)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)

parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--graphflow', type=eval, default=True, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
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
    "--max_grad_norm", type=float, default=5,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/graph_generation/graph_flow")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--sample_freq", type=int, default=40)
parser.add_argument("--log_freq", type=int, default=10)

''' graph generation arguments '''
parser.add_argument("--graph_type", type=str, default='citeseer_small')
parser.add_argument("--max_prev_node", type=int, default=-1)
parser.add_argument("--test_total_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=4)

''' output results save '''
parser.add_argument("--result_dir", type=str, default='graph_gen')

args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)

# define graph files
args.fname          = '_' + args.model_name + '_' + args.graph_type + '_'
args.fname_pred     = '_' + args.model_name + '_' + args.graph_type + '_pred_'
args.fname_train    = '_' + args.model_name + '_' + args.graph_type + '_train_'
args.fname_test     = '_' + args.model_name + '_' + args.graph_type + '_test_'


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def get_loaders(args):
    graphs = create_graphs.create(args)

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_validate = graphs[0:int(0.2*graphs_len)]


    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)


    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])


    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))


    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.result_dir + args.fname_train + '0.dat')
    save_graph_list(graphs, args.result_dir + args.fname_test  + '0.dat')
    print('train and test graphs saved at: ', args.result_dir  + args.fname_test + '0.dat')


    ### dataset initialization
    train_dataset = DualGraph_sampler_flow(graphs_train, 
                                           max_num_node=args.max_num_node)
    test_dataset  = DualGraph_sampler_flow(graphs_test, 
                                           max_num_node=args.max_num_node)
    aaa = train_dataset.__getitem__(1)

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers)
    
    return train_loader, test_loader


def add_noise(x, K):
    """
    [0, 1] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * (K - 1) + noise
    x = x / K
    return x


def compute_bits_per_dim(graphs, graph_matrix, graph_mask, graph_size, flow_model, args):
    # variational dequantization
    if args.vd:
        graphs, logqu = flow_model.module.dequantizer(graphs, 2)
    elif args.add_noise:
        graphs = add_noise(graphs, 2)

    zero = torch.zeros(graphs.shape[0], 1).to(graphs)

    flow_model.module.set_masks(graph_mask)
    flow_model.module.set_E(graph_matrix)

    z, delta_logp = flow_model(graphs, zero)

    if not args.require_prior:
        log_pz = standard_normal_logprob(z).sum(-1) * graph_mask
    else:
        graph_K      = (graph_mask * graph_mask.sum(1, keepdim=True)).expand(z.shape[:-1])
        log_pz       = flow_model.module.get_logp(z, graph_K.long()).sum(-1) * graph_mask

    log_pz  = log_pz.sum(1, keepdim=True)
    nlog_px = - (log_pz - delta_logp)
  
    if args.vd:
        nlog_px = nlog_px + logqu

    return nlog_px.mean()


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))
    
    def build_cnf():
        graphode_diffeq = layers.ODEGraphnetGraphGen(
            hidden_dims=hidden_dims,
            input_shape=data_shape,
            strides=strides,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            num_squeeze=0,
            ifgate=False,
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

    # define the flow model
    chain = []
    if args.use_logit:
        chain = [layers.LogitTransform(alpha=args.alpha)]
    chain = chain + [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        chain.append(layers.MovingBatchNorm2d(data_shape[0]))
    data_nelements = 1
    for ele in data_shape:
      data_nelements *= ele
    flow_model = layers.SequentialFlow(chain, \
                                       prior_config=[args.require_prior, (args.max_num_node - 1) * args.max_num_node, 1], \
                                       variational_dequantization=args.vd)
    return flow_model


def decode_to_adj(up_tri_mat, edges, num_edges, num_nodes):
    up_tri_mat = up_tri_mat.cpu().data.numpy()
    adjs = []

    for i in range(0, up_tri_mat.shape[0]):
        adj = np.zeros((int(num_nodes[i]), int(num_nodes[i])))
        for j in range(0, int(num_edges[i])):
            r,c      = edges[i][j]
            adj[r,c] = int(up_tri_mat[i][j] > 0.5)
            adj[c,r] = int(up_tri_mat[i][j] > 0.5)
        adjs += [adj]

    return adjs


if __name__ == "__main__":

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, test_loader = get_loaders(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    data_shape = (1,)
    flow_model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(flow_model, logger)
    set_cnf_options(args, flow_model)
    logger.info(flow_model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(flow_model)))

    # optimizer
    flow_optimizer = optim.Adam(flow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        flow_model.load_state_dict(checkpt["flow_state_dict"])
        if "optim_state_dict" in checkpt.keys():
            flow_optimizer.load_state_dict(checkpt["flow_optim_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    if torch.cuda.is_available():
        flow_model = torch.nn.DataParallel(flow_model).cuda()

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    flow_grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(flow_model, 500)

    best_loss = float("inf")
    train_itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        flow_model.train()
        for _, ( graphs, graph_matrix, graph_masks, _, graph_size ) in enumerate(train_loader):
            start = time.time()
            current_lr = update_lr(flow_optimizer, train_itr)
            flow_optimizer.zero_grad()
        
            # cast data and move to device
            graphs, graph_matrix, graph_masks = [cvt(x) for x in [graphs, graph_matrix, graph_masks]]

            # compute loss
            loss = compute_bits_per_dim(graphs, 
                                        graph_matrix, 
                                        graph_masks, 
                                        graph_size,
                                        flow_model, 
                                        args)

            if regularization_coeffs:
                reg_states = get_regularization(flow_model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            total_time = count_total_time(flow_model)
            loss = loss + total_time * args.time_penalty
        
            loss.backward()
            flow_grad_norm = torch.nn.utils.clip_grad_norm_(flow_model.parameters(), args.max_grad_norm)
        
            flow_optimizer.step()
        
            if args.spectral_norm: spectral_norm_power_iteration(flow_model, args.spectral_norm_niter)
        
            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(flow_model))
            flow_grad_meter.update(flow_grad_norm)
            tt_meter.update(total_time)
        
            #if itr % args.log_freq == 0:
            if train_itr % 1 == 0:
                proxy_optimizer = flow_optimizer
                   
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Flow Grad Norm {:.4f}({:.4f}) | "
                    "Total Time {:.2f}({:.2f}) | lr {:.6f}".format(
                        train_itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                        steps_meter.avg, flow_grad_meter.val, flow_grad_meter.avg, 
                        tt_meter.val, tt_meter.avg, current_lr
                    )
                )
                if regularization_coeffs:
                  log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)
        
            train_itr += 1

        # compute test loss
        flow_model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = 0
                itr = 0
                num_batches = 0
                for ( graphs, graph_matrix, graph_masks, _, graph_size ) in test_loader:
                    graphs, graph_matrix, graph_masks = [cvt(x) for x in [graphs, graph_matrix, graph_masks]]
                    loss = compute_bits_per_dim(graphs, 
                                                graph_matrix, 
                                                graph_masks, 
                                                graph_size,
                                                flow_model, 
                                                args)

                    num_batches += 1
                    losses += loss
                    itr += 1

                loss = losses / num_batches
                logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    dict_to_save = {"args": args}
                    dict_to_save['flow_state_dict'] = flow_model.module.state_dict() if torch.cuda.is_available() else flow_model.state_dict()
                    dict_to_save['flow_optim_state_dict'] = flow_optimizer.state_dict()
                    torch.save(dict_to_save, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        if epoch % args.sample_freq == 0 or epoch == 1:
            # save the models
            dict_to_save = {"args": args}
            dict_to_save['flow_state_dict']       = flow_model.module.state_dict() if torch.cuda.is_available() else flow_model.state_dict()
            dict_to_save['flow_optim_state_dict'] = flow_optimizer.state_dict()
            torch.save(dict_to_save, os.path.join(args.result_dir, "checkpt_" + str(epoch) + ".pth"))

            # perform sampling
            G_pred = []
            G_gt   = []
            with torch.no_grad():
                total_size_len = 0
                total_size_count = 0 
                test_total_size_len = 0
                test_total_size_count = 0 
                test_iter = iter( test_loader )
                while len(G_pred) < args.test_total_size:
                    try:
                      fixed_graphs, fixed_graph_matrix, fixed_mask, fixed_nodes, graph_sizes = next(test_iter)
                    except:
                      test_iter = iter( test_loader )
                      fixed_graphs, fixed_graph_matrix, fixed_mask, fixed_nodes, graph_sizes = next(test_iter)
                    total_size_count += graph_sizes.sum()
                    total_size_len += len(graph_sizes)
                    fixed_graphs, fixed_graph_matrix, fixed_mask = [cvt(x) for x in [fixed_graphs, fixed_graph_matrix, fixed_mask]]

                    flow_model.module.set_masks(fixed_mask)
                    flow_model.module.set_E(fixed_graph_matrix)

                    if not args.require_prior:
                        fixed_z    = cvt(torch.randn(fixed_graphs.shape[0], fixed_graphs.shape[1], *data_shape))
                    else:
                        fixed_z    = cvt(flow_model.module.get_prior_samples(fixed_mask.sum(1, keepdim=True).expand(fixed_graphs.shape[:-1]).long()))

                    up_tri_mat = flow_model(fixed_z, reverse=True)
                    adj_generated_data = decode_to_adj(up_tri_mat, fixed_nodes, fixed_mask.sum(1), graph_sizes)
                    adj_gt_data        = decode_to_adj(fixed_graphs, fixed_nodes, fixed_mask.sum(1), graph_sizes)
                 
                    G_pred_list = []
                    G_gt_list   = []
                    for i in range(0, len(adj_generated_data)):
                        G_i    = get_graph(adj_generated_data[i]) # get a graph from zero-padded adj
                        G_gt_i = get_graph(adj_gt_data[i])
                        G_pred_list.append(G_i)
                        G_gt_list.append(G_gt_i)
                        test_total_size_len += G_gt_i.number_of_nodes()
                        test_total_size_count += 1
                 
                    G_pred += G_pred_list
                    G_gt   += G_gt_list
                 
                # save results
                pred_fname = os.path.join(args.result_dir, args.model_name + '_' + args.graph_type + \
                                                                       '_epoch-' + str(epoch) + '_pred.dat')
                gt_fname   = os.path.join(args.result_dir, args.model_name + '_' + args.graph_type + \
                                                                       '_epoch-' + str(epoch) + '_gt.dat')
                save_graph_list(G_pred, pred_fname)
                save_graph_list(G_gt, gt_fname)
                doublecheck_len   = 0
                for ele in G_gt:
                  doublecheck_len += ele.number_of_nodes()
                print('In total:', float(doublecheck_len) / len(G_gt))
                print('Finished sampling')
