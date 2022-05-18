#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset_ddp import CPDataset, CPDataLoader
from networks_ddp import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images

# dist imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

# time it
import timeit

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    # dist flags
    parser.add_argument("--distributed", action='store_true', help='run on distributed')

    opt = parser.parse_args()
    return opt

########## DDP SETUP ############

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group(dist.Backend.NCCL, init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# - added utility method found on here https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
# - basically this method adds optimization params to cuda since method is not native on Pytorch
# - turns out that so far DistributedDataParallel is not complaining without using this method but it is here just in case
# - add it right after defining optimizer before train loop 
def optimizer_to(optim):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.cuda()
            if param._grad is not None:
                param._grad.data = param._grad.data.cuda()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.cuda()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.cuda()

#################################

def train_gmm(opt, train_loader, model, board, rank, world_size, batch_size):
    # distributed
    if opt.distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model, device_ids=[rank])

    # set model to train
    model.train()

    # criterion
    criterionL1 = nn.L1Loss().cuda()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    #optimizer_to(optimizer)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    print('dataset length  :' + str(len(train_loader.dataset)))
    print('loader length  :' + str(len(train_loader)))
    # - define epoch lengths for DDP to call set_epoch method
    # - we are calling drop_last in dataloader to ensure that 
    # these lengths are consistent for all batches
    # - note that if dataset size is divisible by batches BUT
    # result is not divisble by number of GPUs, random samples
    # will be dropped in a similar way as with drop_last
    # - it turns out that everything is handled accordingly with this code
    epoch_length = len(train_loader.dataset) // batch_size # number of steps needed for a single GPU to cover all of dataset
    shared_epoch_length = epoch_length // world_size # number of steps needed for all GPUs to cover all of dataset
    
    if rank == 0:
        epochs = int(np.ceil((opt.keep_step + opt.decay_step) / epoch_length))
        shared_epochs = int(np.ceil((opt.keep_step + opt.decay_step) / shared_epoch_length))
        print('Original dataset size: ' + str(len(train_loader.dataset)))
        print('Trimmed dataset size: ' + str(epoch_length * batch_size))
        print('Batch size: ' + str(batch_size))
        print('Epoch length: ' + str(epoch_length))
        print('Shared epoch length: ' + str(shared_epoch_length))
        print('No. of epochs: ' + str(epochs)) # number of epochs
        print('No. of shared epochs: ' + str(shared_epochs)) # number of shared epochs needed to train as much as number of epochs
        print('No. of steps needed to get to train as much as no. of epochs: ' +  str(shared_epoch_length * epochs))

    for step in range(opt.keep_step + opt.decay_step):
        # get current epoch by getting step as multiple of shared_epoch_length
        epoch = step // shared_epoch_length
        np.random.seed(epoch)
        random.seed(epoch)

        if opt.distributed:
            # fix sampling seed such that each gpu gets different part of dataset
            train_loader.data_loader.sampler.set_epoch(epoch)

        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # load all images onto cuda
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        
        # debug distributed data loading
        #print('Current step: ' + str(step), 'Current epoch: ' + str(epoch), 'Running on rank: ' + str(rank), 'Index: ' + str(inputs['index']))
            
        # compute TPS params for warping
        grid, theta = model(agnostic, c)

        # warp the clothing
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        # set visuals for tensorboard
        visuals = [ [im_h, shape, im_pose], 
                [c, warped_cloth, im_c], 
                [warped_grid, (warped_cloth+im)*0.5, im]]
            
        # get warp loss
        loss = criterionL1(warped_cloth, im_c)
            
        # backward step
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
                
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('L1 Loss', loss.item(), step+1)
            t = time.time() - iter_start_time
            if rank == 0:
                print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def train_tom(opt, train_loader, model, board, rank, world_size, batch_size):
    # set model to train
    model.train()

    # put model on cuda rank
    model.cuda()
    
    # criterion
    criterionL1 = nn.L1Loss().cuda()
    criterionVGG = VGGLoss().cuda()
    criterionMask = nn.L1Loss().cuda()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    #optimizer_to(optimizer, rank)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    # adding for multiple GPUs    
    if opt.distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model, device_ids=[rank])

    # - define epoch lengths for DDP to call set_epoch method
    # - we are calling drop_last in dataloader to ensure that 
    # these lengths are consistent for all batches
    epoch_length = len(train_loader.dataset) // batch_size # number of steps needed for a single GPU to cover all of dataset
    shared_epoch_length = epoch_length // world_size # number of steps needed for all GPUs to cover all of dataset
    
    if rank == 0:
        epochs = int(np.ceil((opt.keep_step + opt.decay_step) / epoch_length))
        shared_epochs = int(np.ceil((opt.keep_step + opt.decay_step) / shared_epoch_length))
        print('Original dataset size: ' + str(len(train_loader.dataset)))
        print('Trimmed dataset size: ' + str(epoch_length * batch_size))
        print('Batch size: ' + str(batch_size))
        print('Epoch length: ' + str(epoch_length))
        print('Shared epoch length: ' + str(shared_epoch_length))
        print('No. of epochs: ' + str(epochs)) # number of epochs
        print('No. of shared epochs: ' + str(shared_epochs)) # number of shared epochs needed to train as much as number of epochs
        print('No. of steps needed to get to train as much as no. of epochs: ' +  str(shared_epoch_length * epochs))
    
    for step in range(opt.keep_step + opt.decay_step):
        # get current epoch by getting step as multiple of shared_epoch_length
        epoch = step // shared_epoch_length
        np.random.seed(epoch)
        random.seed(epoch)

        if opt.distributed:
            # fix sampling seed such that each gpu gets different part of dataset
            train_loader.data_loader.sampler.set_epoch(epoch)

        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # debug distributed data loading
        print('Current step: ' + str(step), 'Current epoch: ' + str(epoch), 'Running on rank: ' + str(rank), 'Index: ' + str(inputs['index']))
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            if rank == 0:
                print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def main(rank, world_size, opt):
    # is cuda being used?
    #print(torch.cuda.is_available())
    #print(torch.cuda.current_device())
    #print(torch.cuda.device(0))
    #print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(mp.cpu_count())

    if rank == 0:
        print(opt)
        print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # ensure that each process exclusively works on a single GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    batch_size = opt.batch_size

    if opt.distributed:
        # setup distributed environment
        setup(rank, world_size)

    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset, rank, world_size)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt).to(device)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            print('Loading checkpoint!')
            load_checkpoint(model, opt.checkpoint)
        if rank == 0:
            starttime = timeit.default_timer()
            print('Timer started')
        train_gmm(opt, train_loader, model, board, rank, world_size, batch_size)
        if rank == 0:
            print("Time taken : ", timeit.default_timer() - starttime)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        if rank == 0:
            starttime = timeit.default_timer()
            print('Timer started')
        train_tom(opt, train_loader, model, board, rank, world_size, batch_size)
        if rank == 0:
            print("Time taken : ", timeit.default_timer() - starttime)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
    
    if opt.distributed:
        # destroy process group
        cleanup()

    if rank == 0:
        print('Finished training %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":    
    opt = get_opt()

    if opt.distributed:
        world_size = torch.cuda.device_count()

        mp.spawn(
            main,
            args=(world_size, opt),
            nprocs=world_size
        )
    else:
        # run method normally as rank 0 with 1 GPU (world_size = 1)
        main(0, 1, opt)