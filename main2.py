#######################################################################################################################
#
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
# ********************************************************************************************************************
#
#
# The code in this file is adapted from: https://github.com/pytorch/examples/tree/master/imagenet/main.py
#
# Main Difference from the original file: add the networks using partial convolution based padding
#
# Network options using zero padding:               vgg16_bn, vgg19_bn, resnet50, resnet101, resnet152, ... 
# Network options using partial conv based padding: pdvgg16_bn, pdvgg19_bn, pdresnet50, pdresnet101, pdresnet152, ...
#
# Contact: Guilin Liu (guilinl@nvidia.com)
#
#######################################################################################################################
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.models as models_baseline # networks with zero padding
import models as models_partial # partial conv based padding 


model_baseline_names = sorted(name for name in models_baseline.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_baseline.__dict__[name]))

model_partial_names = sorted(name for name in models_partial.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_partial.__dict__[name]))

model_names = model_baseline_names + model_partial_names


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--data_test', metavar='DIRTEST',
                    help='path to test dataset')
# parser.add_argument('--data_train', metavar='DIRTRAIN',
#                     help='path to training dataset')

# parser.add_argument('--data_val', metavar='DIRVAL',
#                     help='path to validation dataset')                    

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')



parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 192)')


parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--ckptdirprefix', default='', type=str)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    checkpoint_dir = args.ckptdirprefix + 'checkpoint_' + args.arch + '_' + args.prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.logger_fname = os.path.join(checkpoint_dir, 'loss.txt')

    with open(args.logger_fname, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)    
        log_file.write('world size: %d\n' % args.world_size)
		
		
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch](pretrained=True)
        else:
            model = models_partial.__dict__[args.arch](pretrained=True)
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch]()
        else:
            model = models_partial.__dict__[args.arch]()
        # model = models.__dict__[args.arch]()


    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('model created\n')
		
		
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        if args.arch.startswith('alexnet') or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    test_dir = args.data_test  # os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))     

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dir, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('training/val dataset created\n')



    # logging
    with open(args.logger_fname, "a") as log_file:
        log_file.write('started training\n')

    for epoch in range(1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        test(test_loader, model ,epoch)

    
    
    
    

def test(train_loader, model, epoch):

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):


        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)


        output = model(input)


if __name__ == '__main__':
    main()
