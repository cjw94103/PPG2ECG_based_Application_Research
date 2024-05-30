import numpy as np
import torch
import argparse
import os

from dataset import PPG_Dataset
from models import vgg16_bn
from gan_models import GeneratorResNet
from train_func import train_PPGECG_model
from make_args import Args

# for reproductivity
seed = 1994
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/VGG16_PPGECG.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

# load partition
partition = np.load(args.partition_path, allow_pickle=True).item()

trainset = partition['trainset']
valset = partition['valset']

# get data generator
def get_dataloader(batch_size, num_workers):
    trainloader_instance = PPG_Dataset(filepaths=trainset, sampling_rate=args.target_sampling_rate, 
                                       min_max_norm=args.min_max_norm, z_score_norm=args.z_score_norm, interp=args.interp_method)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainloader_instance, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(trainloader_instance,
                                                   batch_size = batch_size,
                                                   shuffle = None,
                                                   num_workers = num_workers,
                                                   drop_last = True,
                                                   pin_memory = True,
                                                   sampler = train_sampler)
    
    valloader_instance = PPG_Dataset(filepaths=valset, sampling_rate=args.target_sampling_rate, 
                                     min_max_norm=args.min_max_norm, z_score_norm=args.z_score_norm, interp=args.interp_method)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valloader_instance, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valloader_instance,
                                                 batch_size = batch_size,
                                                 shuffle = None,
                                                 num_workers = num_workers,
                                                 drop_last = True,
                                                 pin_memory = True,
                                                 sampler=val_sampler)
    
    return train_dataloader, val_dataloader

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
    base_lr = args.lr
    weight_decay = args.weight_decay

    # init process group
    batch_size = int(batch_size / world_size)
    num_worker = int(num_worker / world_size)
    setup(rank, world_size)

    # load P2E GAN
    weights_path = './gan_weights/PPG2ECG_CycleGAN_1Epochs.pth'
    input_shape = (None, 1, int(args.target_sampling_rate * args.sig_time_len))
    n_residual_blocks = 6
    
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    weights = torch.load(weights_path)
    G_AB.load_state_dict(weights['G_AB'])
    G_AB.to(rank)
    G_AB.eval()

    # load model
    cls_model = vgg16_bn(in_channels=args.in_channels, num_classes=args.num_classes)
    cls_model = cls_model.to(rank)
    cls_model = torch.nn.parallel.DistributedDataParallel(cls_model, device_ids=[rank]) 

    # get dataloder
    train_dataloader, val_dataloader = get_dataloader(batch_size, num_worker)

    # get optimizer
    optimizer = torch.optim.Adam(cls_model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Learning
    train_PPGECG_model(model=cls_model,
                       G_AB=G_AB,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       z_score_norm=args.z_score_norm_rescale,
                       min_max_norm=args.min_max_norm_rescale,
                       num_epochs=num_epochs,
                       optimizer=optimizer,
                       base_lr=base_lr,
                       monitor=args.monitor,
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