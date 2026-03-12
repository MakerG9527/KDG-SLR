import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

def _optimizer(config, model, fusion_model, model_text):
    if isinstance(config.solver.t_ratio, dict):
        t_ratio = config.solver.t_ratio.value if 'value' in config.solver.t_ratio else 1.0
    else:
        t_ratio = config.solver.t_ratio

    if isinstance(config.solver.mamba_ratio, dict):
        mamba_ratio = config.solver.mamba_ratio.value if 'value' in config.solver.mamba_ratio else 1.0
    else:
        mamba_ratio = config.solver.mamba_ratio

    # 🌟 修复：处理 vision_ratio 或 ratio 参数
    # 确保它是一个浮点数，以便进行乘法运算
    if hasattr(config.solver, 'ratio'):
        if isinstance(config.solver.ratio, dict):
            ratio = config.solver.ratio.value if 'value' in config.solver.ratio else 1.0
        else:
            ratio = config.solver.ratio
    else:
        # 如果没有 ratio 参数，则默认为 1.0
        ratio = 1.0

    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters()},
                                {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
                                {'params': model_text.parameters(), 'lr': config.solver.lr * mamba_ratio}],
                               lr=config.solver.lr, betas=(0.9, 0.98), eps=1e-8,
                               weight_decay=0.2)
        print('Adam')
    elif config.solver.optim == 'sgd':
        optimizer = optim.SGD([{'params': model.parameters()},
                               {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
                               {'params': model_text.parameters(), 'lr': config.solver.lr * mamba_ratio}],
                              lr=config.solver.lr, momentum=0.9, weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        # 🌟 修复：这里 `model` 的参数组可能需要一个 `ratio` 参数
        # 你的 `train.py` 报错是在这一行，所以需要修正
        optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.solver.lr * ratio},
                                 {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},
                                 {'params': model_text.parameters(), 'lr': config.solver.lr * mamba_ratio}],
                                betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer

def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step,
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step,
            gamma=config.solver.lr_decay_rate
        )
    else:
        raise ValueError("Unknown lr scheduler type: {}".format(config.solver.type))
    return lr_scheduler