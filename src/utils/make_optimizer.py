import transformers
from torch import optim


class ValLossDecayScheduler:

    def __init__(self, optimizer, warmup_steps, val_loss_decay_time):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.now_step = 0
        self.val_loss_decay = True
        self.init_lr = self.optimizer.param_groups[0]['lr']
        self.val_loss_decay_time = val_loss_decay_time
        if warmup_steps != 0:
            for param in self.optimizer.param_groups:
                param['lr'] = 0.

    def step(self):
        self.now_step += 1
        if self.now_step < self.warmup_steps:
            for param in self.optimizer.param_groups:
                param['lr'] = self.now_step / self.warmup_steps * self.init_lr


class StepDecayScheduler:

    def __init__(self, optimizer, warmup_steps, decay_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.now_step = 0
        self.decay_steps = decay_steps
        self.init_lr = self.optimizer.param_groups[0]['lr']
        if warmup_steps != 0:
            for param in self.optimizer.param_groups:
                param['lr'] = 0.

    def step(self):
        self.now_step += 1
        if self.now_step < self.warmup_steps:
            for param in self.optimizer.param_groups:
                param['lr'] = self.now_step / self.warmup_steps * self.init_lr
        else:
            if (self.now_step - self.warmup_steps) % self.decay_steps == 0 and self.now_step != self.warmup_steps:
                old_lr = self.optimizer.param_groups[0]['lr']
                new_lr = 0.5 * old_lr
                for param in self.optimizer.param_groups:
                    param['lr'] = new_lr


def make_optimizer(name, model, learning_rate, decay_type, num_cycles, decay_steps, val_loss_decay_time,
                   warmup_steps, total_steps, beta_1, beta_2, epsilon,
                   weight_decay):
    assert name in ['adam', 'adamw', 'sgd']
    if name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay)
    elif name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=beta_1)
    assert decay_type in [
        'linear_decay', 'consine_decay',
        'consine_restart_decay', 'val_loss_decay', 'step_decay', None
    ]
    if decay_type == 'linear_decay':
        schedule = transformers.get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)
    elif decay_type == 'consine_decay':
        schedule = transformers.get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)
    elif decay_type == 'consine_restart_decay':
        schedule = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, num_cycles)
    elif decay_type == 'val_loss_decay':
        schedule = ValLossDecayScheduler(
            optimizer, warmup_steps, val_loss_decay_time)
    elif decay_type == 'step_decay':
        schedule = StepDecayScheduler(optimizer, warmup_steps, decay_steps)
    return optimizer, schedule

