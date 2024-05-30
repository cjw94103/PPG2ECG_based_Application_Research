import torch
import torch.nn as nn

import numpy as np
import os
import itertools
import argparse

from dataset import PPG2ECG_Dataset
from models import Discriminator, GeneratorResNet, weights_init_normal
from train_func import train_CycleGAN
from make_args import Args

# for reproductivity
torch.manual_seed(777)
np.random.seed(777)

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/CycleGAN_PPG2ECG.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

# load partition
partition = np.load(args.partition_path, allow_pickle=True).item()

trainset = partition['trainset']
valset = partition['valset']

# get_dataloader function
def get_dataloader(batch_size, num_worker):
    trainloader_instance = PPG2ECG_Dataset(trainset, trainset, sampling_rate=args.target_sampling_rate, 
                                           min_max_norm=args.min_max_norm, z_score_norm=args.z_score_norm, interp=args.interp_method)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainloader_instance, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(trainloader_instance,
                                                   batch_size = batch_size,
                                                   shuffle = None,
                                                   num_workers = num_worker,
                                                   drop_last = True,
                                                   pin_memory = True,
                                                   sampler = train_sampler)
    
    return train_dataloader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.dist_proc_port
    
    torch.distributed.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    num_worker = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_per_epochs = args.save_per_epochs

    # optimizer parameter
    b1 = args.b1
    b2 = args.b2
    lr_decay_epoch = args.lr_decay_epoch

    # model parameter
    input_shape = (None, 1, int(args.target_sampling_rate * args.sig_time_len))
    n_residual_blocks = args.n_residual_blocks
    D_output_shape = (batch_size, 1, (input_shape[-1] // 2 ** 4) + 1)

    # init process group
    batch_size = int(batch_size / world_size)
    num_worker = int(num_worker / world_size)
    setup(rank, world_size)

    # loss and lr parameter
    lambda_cyc = args.lambda_cyc
    base_lr = args.lr

    # load model
    G_AB = GeneratorResNet(input_shape, n_residual_blocks).to(rank)
    G_AB = torch.nn.parallel.DistributedDataParallel(G_AB, device_ids=[rank])

    G_BA = GeneratorResNet(input_shape, n_residual_blocks).to(rank)
    G_BA = torch.nn.parallel.DistributedDataParallel(G_BA, device_ids=[rank])

    D_A = Discriminator(input_shape).to(rank)
    D_A = torch.nn.parallel.DistributedDataParallel(D_A, device_ids=[rank])

    D_B = Discriminator(input_shape).to(rank)
    D_B = torch.nn.parallel.DistributedDataParallel(D_B, device_ids=[rank])

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # get generator
    train_dataloader = get_dataloader(batch_size, num_worker)

    # get optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=base_lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=base_lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=base_lr, betas=(b1, b2))

    # Learning
    train_CycleGAN(G_AB=G_AB, 
                   G_BA=G_BA, 
                   D_A=D_A, 
                   D_B=D_B, 
                   train_dataloader=train_dataloader, 
                   lambda_cyc=lambda_cyc, 
                   D_output_shape=D_output_shape,
                   num_epochs=num_epochs, 
                   early_stop_patience=None,
                   optimizer_G=optimizer_G, 
                   optimizer_D_A=optimizer_D_A, 
                   optimizer_D_B=optimizer_D_B, 
                   lr_decay_epoch=lr_decay_epoch, 
                   base_lr=base_lr, 
                   monitor=None, 
                   save_per_epochs=save_per_epochs, 
                   model_save_path=args.model_save_path, 
                   multi_gpu_flag=True,
                   DEVICE=rank)
    # clean process
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, ), join=True)
    
if __name__ == '__main__':
    main()