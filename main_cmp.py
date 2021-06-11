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
import torch.optim
from torch.autograd import Variable

import threading
from queue import Queue

import models
from utils import AverageMeter, ProgressMeter, Accuracy
from datasets import GetDataset

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='/export/Data/cifar', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='(cifar10, cifar100, imagenet)')
parser.add_argument('-a', '--arch', default='preresnet164_split',
                    help='model architecture (default: preresnet164_split)')
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

parser.add_argument('--period', type=int, default=0, help='averaging period')


def create_model(args):
    if args.dataset == 'cifar10':
        blocks = models.__dict__[args.arch](num_classes=10, slices=args.model_slices)
    elif args.dataset == 'cifar100':
        blocks = models.__dict__[args.arch](num_classes=100, slices=args.model_slices)
    elif args.dataset == 'imagenet':
        blocks = models.__dict__[args.arch](slices=args.model_slices)
    else:
        assert False
    return blocks


def get_model_size(model_blocks):
    n_params = 0
    for block in model_blocks:
        n_params += np.sum([p.numel() for p in block.parameters()])
    return n_params * 8 / 1024 ** 3


def main():
    args = parser.parse_args()
    print(args, flush=True)

    args.ngpus = torch.cuda.device_count()
    assert args.ngpus > 0
    assert len(args.model_slices) == args.ngpus

    # Create model blocks on GPU
    print("=> creating model '{}'".format(args.arch))
    models = [create_model(args) for _ in range(2)]
    print('model size: {:.2f} GiB'.format(get_model_size(models[0])))
    for block_idx in range(args.ngpus):
        # sync models
        for k, v in models[0][block_idx].state_dict().items():
            models[1][block_idx].state_dict()[k].copy_(v.data)

        models[0][block_idx] = models[0][block_idx].cuda(block_idx)
        models[1][block_idx] = models[1][block_idx].cuda(args.ngpus - 1 - block_idx)

    # define loss function (criterion) and optimizer
    criterions = [
        nn.CrossEntropyLoss().cuda(args.ngpus - 1),
        nn.CrossEntropyLoss().cuda(0),
    ]
    optimizers = [[torch.optim.SGD(
        block.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    ) for block in models[i]] for i in range(2)]

    # Data loading code
    train_dataset, val_dataset = GetDataset(args.dataset, args.data)
    samplers = [torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=2, rank=i) for i in range(2)]
    train_loaders = [torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=samplers[i], num_workers=args.workers, pin_memory=True) for i in range(2)]
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Control the flow in threads
    barrier = threading.Barrier(2)
    qs = [Queue(maxsize=args.ngpus) for _ in range(2)]
    for _ in range(args.ngpus):
        qs[0].put(0)

    # Train and eval
    print('=> start training...')
    for epoch in range(args.epochs):
        ths = [threading.Thread(
            target=train_epoch_thread,
            args=(thread_id, train_loaders, models, criterions, optimizers,
                  epoch, args, qs, barrier),
            daemon=True,
        ) for thread_id in range(2)]
        ths[0].start()
        ths[1].start()
        ths[0].join()
        ths[1].join()

        if not args.noeval:
            validate(0, val_loader, models[0], criterions[0], epoch, args)
            validate(1, val_loader, models[1], criterions[1], epoch, args)


def avg(thread_id, block_id, models, cur_step, lr, args):
    if args.period and (cur_step + 1) % args.period == 0:
        device = block_id if thread_id == 0 else len(models[0]) - 1 - block_id

        my_block = models[thread_id][block_id]
        block = models[1 - thread_id][block_id]

        for my_p, p in zip(my_block.parameters(), block.parameters()):
            #my_p.data.add_(p.data.cuda(device)).div_(2)
            if thread_id == 0:
                my_p.data.add_(p.data.cuda(device)).div_(2)
            else:
                my_p.data.copy_(p.data).add_(- lr / 2, my_p.grad)


def train_epoch_thread(thread_id, train_loaders, models, criterions, optimizers,
                       epoch, args, qs, barrier):
    epoch_time = AverageMeter('Time', ':9.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.epochs,
        [epoch_time, losses, top1, top5],
        prefix='[Thread {}] Train: Epoch '.format(thread_id),
    )

    if thread_id == 0:
        devices = [i for i in range(args.ngpus)]
    else:
        devices = [i for i in range(args.ngpus - 1, -1, -1)]

    my_train_loader = train_loaders[thread_id]
    my_model_blocks = models[thread_id]
    my_criterion = criterions[thread_id]
    my_optimizers = optimizers[thread_id]

    for block in my_model_blocks:
        block.train()

    barrier.wait()
    start = time.time()
    for batch_idx, (images, targets) in enumerate(my_train_loader):
        if args.epoch_batches and batch_idx == args.epoch_batches:
            break
        cur_step = batch_idx + len(my_train_loader) * epoch
        lr = adjust_learning_rate(my_optimizers, epoch, batch_idx, len(my_train_loader), args)

        images = images.cuda(devices[0], non_blocking=True)
        targets = targets.cuda(devices[-1], non_blocking=True)

        for optimizer in my_optimizers:
            optimizer.zero_grad()

        """ forward """
        checkpoints = [[], []]

        inputs = images
        for block_id in range(args.ngpus):
            qs[thread_id].get()

            block = my_model_blocks[block_id]
            inputs = block(inputs.cuda(devices[block_id]))

            if block_id != args.ngpus - 1:
                checkpoints[0].append(inputs)
                inputs = inputs.detach()
                inputs = Variable(inputs, requires_grad=True)
                checkpoints[1].append(inputs)
            else: # last block compute loss
                loss = my_criterion(inputs, targets)
                acc1, acc5 = Accuracy(inputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        if args.show_mem:
            mems = [torch.cuda.memory_allocated(i) / 1024 ** 3 for i in range(args.ngpus)]
            print('GPU memory usage: {:.2f} GiB = sum:'.format(np.sum(mems)),
                  ''.join('{:.2f} '.format(mem) for mem in mems))

        """ backward """
        for block_id in range(args.ngpus - 1, -1, -1):
            if block_id == args.ngpus - 1:
                loss.backward()
            else:
                outputs = checkpoints[0][block_id]
                err_grad = checkpoints[1][block_id].grad
                outputs.backward(err_grad)
                del checkpoints[0][block_id]
                del checkpoints[1][block_id]

            my_optimizers[block_id].step()

            avg(thread_id, block_id, models, cur_step, lr, args)
            qs[1 - thread_id].put(1 - thread_id)


    barrier.wait()
    epoch_time.update(time.time() - start, 1)
    progress.display(epoch)


def validate(thread_id, val_loader, model_blocks, criterion, epoch, args):
    epoch_time = AverageMeter('Time', ':9.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.epochs, [epoch_time, losses, top1, top5],
        prefix='[Thread {}] Test: Epoch '.format(thread_id),
    )

    if thread_id == 0:
        devices = [i for i in range(args.ngpus)]
    else:
        devices = [i for i in range(args.ngpus - 1, -1, -1)]

    # evaluate mode
    for block in model_blocks:
        block.eval()

    with torch.no_grad():
        start = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda(devices[0], non_blocking=True)
            targets = targets.cuda(devices[-1], non_blocking=True)

            # compute output
            inputs = images
            for block, device in zip(model_blocks, devices):
                inputs = inputs.cuda(device)
                inputs = block(inputs)
            loss = criterion(inputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = Accuracy(inputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        epoch_time.update(time.time() - start)
        progress.display(epoch)


def adjust_learning_rate(optimizers, epoch, batch_idx, n_batch, args):
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
    else:
        assert False

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
