import os
import logging
from uuid import uuid4
from tqdm.auto import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import higher


class MetaDatasetModule(nn.Module):
    def __init__(self, n=100, d=2, n_inner_opt=3, inner_lr=1e-2):
        super().__init__()

        self.X = nn.Parameter(torch.randn(n, d))
        self.register_buffer('Y', torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]))

        self.inner_loss_fn = nn.BCEWithLogitsLoss()
        self.n_inner_opt = n_inner_opt
        self.inner_lr = inner_lr

    def forward(self, device):
        ## Patch meta parameters.
        self.register_parameter('w', nn.Parameter(torch.randn(self.X.size(-1))))
        self.register_parameter('b', nn.Parameter(torch.randn(1)))

        inner_optimizer = torch.optim.SGD([self.w, self.b], lr=self.inner_lr)
        with higher.innerloop_ctx(self, inner_optimizer,
                                  device=device, copy_initial_weights=False,
                                  track_higher_grads=self.training) as (fmodel, diffopt):
            for _ in range(self.n_inner_opt):
                logits = fmodel.X @ fmodel.w + fmodel.b
                inner_loss = self.inner_loss_fn(logits, self.Y)
                diffopt.step(inner_loss)

            return fmodel.X @ fmodel.w + fmodel.b


def train_batch(accelerator, meta_module, optimizer):
    meta_module.train()

    optimizer.zero_grad()

    logits = meta_module(device=accelerator.device)

    loss = nn.BCEWithLogitsLoss()(logits, meta_module.module.Y)
    loss.backward()

    optimizer.step()

    return { 'loss': loss.item() }


def main(accelerator, log_dir=None,
         lr=1e-2, steps=500, inner_steps=3, inner_lr=1e-2):

    meta_module = MetaDatasetModule(n_inner_opt=inner_steps, inner_lr=inner_lr)

    optimizer = optim.Adam([meta_module.X], lr=lr)

    meta_module, optimizer = accelerator.prepare(meta_module, optimizer)

    for s in tqdm(range(steps)):
        step_metrics = train_batch(accelerator, meta_module, optimizer)

        if accelerator.is_main_process:
            accelerator.save(meta_module.state_dict(), Path(log_dir) / 'model.pt')

            if s % 100 == 0:
                logging.debug({ 'step': s, **step_metrics })


def set_logging(log_dir=None):
    root_dir = Path(log_dir or Path(Path.cwd() / '.log') / f'run-{str(uuid4())[:8]}')
    log_dir = Path(str((root_dir / 'files').resolve()))
    log_dir.mkdir(parents=True, exist_ok=True)

    _CONFIG = {
        'version': 1,
        'formatters': {
            'console': {
                'format': '[%(asctime)s] (%(funcName)s:%(levelname)s) %(message)s',
            },
        },
        'handlers': {
            'stdout': {
                '()': logging.StreamHandler,
                'formatter': 'console',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            '': {
                'handlers': ['stdout'],
                'level': os.environ.get('LOGLEVEL', 'INFO'),
            },
        },
    }

    logging.config.dictConfig(_CONFIG)

    logging.info(f'Files stored in "{log_dir}".')

    def finish_logging():
        pass

    return log_dir, finish_logging


def entrypoint(log_dir=None, **kwargs):
    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = set_logging(log_dir=log_dir) if accelerator.is_main_process else [None, None]
    if accelerator.is_main_process:
        logging.info(f'Working with {accelerator.num_processes} process(es).')

    main(accelerator, **kwargs, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)
