import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    # wandb.login(key="215f5d9c16496a0648e6ac5b2f53fc84d4a4ed5c")
    mode = "offline" if args.wandb_offline else "online"
    # Ensure WANDB_MODE is set in environment for robustness
    os.environ["WANDB_MODE"] = mode 
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
        mode=mode,
        settings=wandb.Settings(console="wrap")
    )


def finish():
    if wandb.run is not None:
        wandb.finish()


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"{name}": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x