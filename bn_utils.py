import torch

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, blocks):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    for block in blocks:
        if not check_bn(block):
            return
        block.train()

    n_blocks = len(blocks)
    momentas = [{} for _ in range(n_blocks)]
    for block, momenta in zip(blocks, momentas):
        block.apply(reset_bn)
        block.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(0, non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)

        for block_id in range(n_blocks):
            block = blocks[block_id]
            momenta = momentas[block_id]
            for module in momenta.keys():
                module.momentum = momentum

            input_var = input_var.cuda(block_id)
            input_var = block(input_var)

        n += b

    for block, momenta in zip(blocks, momentas):
        block.apply(lambda module: _set_momenta(module, momenta))
