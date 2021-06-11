import argparse
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torch.optim

from utils import AverageMeter, ProgressMeter, Accuracy
from datasets import GetDataset
import models

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='/export/Data/cifar', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='(cifar10, cifar100, imagenet)')
parser.add_argument('-a', '--arch', default='preresnet164_mp',
                    help='model architecture (default: preresnet164_mp)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')

### learning rate ###
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('-ls', '--lr-schedule', default='linear', type=str,
                    help='linear, cosine')
parser.add_argument('-ds', '--decay-schedule', type=int, nargs='+', default=[100,150],
                    help='learning rate decaying (*0.1) epochs')
parser.add_argument('--wplr', default=0.1, type=float, help='warm-up learning rate')
parser.add_argument('--warm-up', default=0, type=int, help='learning rate warm-up epochs')

### model slices ###
parser.add_argument('-ms', '--model-slices', type=int, nargs='+', default=[15,15,15,14],
                    help='split model slices to devices')

parser.add_argument('-eb', '--epoch-batches', type=int, default=0, help='#batch per epoch')
parser.add_argument('--noeval', action='store_true', help='do not eval')
parser.add_argument('--show-mem', action='store_true', 
                    help='show gpu mem usage after forward and before backward')

def create_model(args):
    if args.dataset == 'cifar10':
        model = models.__dict__[args.arch](num_classes=10, slices=args.model_slices)
    elif args.dataset == 'cifar100':
        model = models.__dict__[args.arch](num_classes=100, slices=args.model_slices)
    elif args.dataset == 'imagenet':
        model = models.__dict__[args.arch](slices=args.model_slices)
    else:
        assert False
    return model


def main():
    args = parser.parse_args()
    print(args, flush=True)

    # Spawn only ONE PROCESS ON ONE NODE for model parallel!
    args.ngpus = torch.cuda.device_count()
    assert args.ngpus > 0
    assert len(args.model_slices) == args.ngpus

    # Create model on CPU
    print("=> creating model '{}' and distribute to {} GPUs".format(args.arch, args.ngpus),
          '(loss on the last GPU)')
    model = create_model(args)

    # define loss function (criterion, last GPU) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.ngpus - 1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    # Data loading code
    train_dataset, val_dataset = GetDataset(args.dataset, args.data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if not args.noeval:
            validate(val_loader, model, criterion, epoch, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    epoch_time = AverageMeter('Time', ':9.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.epochs, [epoch_time, losses, top1, top5],
                             prefix='Train: Epoch ', end=' | ')

    # switch to train mode
    model.train()

    start = time.time()
    for batch_idx, (images, target) in enumerate(train_loader):
        if args.epoch_batches and batch_idx == args.epoch_batches:
            break
        lr = adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader), args)

        images = images.cuda(0, non_blocking=True)
        target = target.cuda(args.ngpus - 1, non_blocking=True)
        optimizer.zero_grad()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.show_mem:
            mems = [torch.cuda.memory_allocated(i) / 1024 ** 3 for i in range(args.ngpus)]
            print('GPU memory usage: {:.2f} GiB = sum:'.format(np.sum(mems)),
                  ''.join('{:.2f} '.format(mem) for mem in mems))

        # measure accuracy and record loss
        acc1, acc5 = Accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
    epoch_time.update(time.time() - start)
    progress.display(epoch)


def validate(val_loader, model, criterion, epoch, args):
    epoch_time = AverageMeter('Time', ':9.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.epochs, [epoch_time, losses, top1, top5], prefix='Test: Epoch ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(args.ngpus - 1, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = Accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        epoch_time.update(time.time() - start)
        progress.display(epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, batch_idx, n_batch, args):
    lr = args.lr
    if epoch < args.warm_up:
        delta = (args.lr - args.wplr) / args.warm_up / n_batch
        lr = args.wplr + delta * (epoch * n_batch + batch_idx)
    elif args.lr_schedule == 'linear':
        for ds in args.decay_schedule:
            if epoch >= ds:
                lr *= 0.1
    elif args.lr_schedule == 'cosine':
        cur_batch = (epoch - args.warm_up) * n_batch + batch_idx
        total_batch = (args.epochs - args.warm_up) * n_batch
        lr *= 0.5 * (1 + math.cos(math.pi * cur_batch / total_batch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
