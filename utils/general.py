# import matplotlib.pyplot as plt
import os
import yaml
# import torch
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional

def _list(func):
    def wrap(a, **kwargs):
        if isinstance(a, list): retval = func(a, **kwargs)
        elif isinstance(a, tuple): retval = func(list(a), **kwargs)
        else: retval = func([a], **kwargs)
        return retval
    return wrap

@_list
def flatten_list(alist):
    flattened_list = []
    if isinstance(alist, type(np.array([0]))): alist = alist.tolist()
    for element in alist:
        if isinstance(element, list) or isinstance(element, tuple): flattened_list.extend(flatten_list(element))
        else: flattened_list.append(element)
    return flattened_list

def list_convert(alist) -> list:
    if alist is None: alist = []
    elif not isinstance(alist, list): alist = [alist]
    return alist

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    return yaml.safe_load(open(Path(file), errors='ignore'))

def yaml_save(file='opt.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)

def increment_path(path, overwrite: bool = False, sep: str = '', mkdir: bool = False):
    path = Path(path).absolute()
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        n = 2
        while True:
            p = f'{path}{sep}{n}{suffix}' 
            if not os.path.exists(p): 
                if overwrite: p = f'{path}{sep}{n-1}{suffix}'
                break
            n += 1

        path = Path(p)

    if mkdir: path.mkdir(parents=True, exist_ok=True)  

    return path

def convert_seconds(seconds: int) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60)

    time_units = []
    if hours > 0: time_units.append(f"{hours}h")
    if minutes > 0: time_units.append(f"{minutes}m")
    if seconds > 0: time_units.append(f"{seconds}s")
    if len(time_units) == 0: time_units.append(f"{seconds}s")
    return "".join(time_units)

def list_uniqifier(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def set_seed(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:
    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning_lite.utilities.seed.pl_worker_init_function`.
    """
    """ 
    Set random seed 
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            print(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = random.randint(min_seed_value, max_seed_value)
                print(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = random.randint(min_seed_value, max_seed_value)

    # print(f"Global seed set to {seed}", _get_rank())
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # for Multi-GPU, exception safe
    tf.random.set_seed(seed)
    

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

    # def _get_rank(
    #     strategy: Optional["lightning.fabric.strategies.Strategy"] = None,
    # ) -> Optional[int]:
    #     if strategy is not None:
    #         return strategy.global_rank
    #     # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    #     # therefore LOCAL_RANK needs to be checked first
    #     rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    #     for key in rank_keys:
    #         rank = os.environ.get(key)
    #         if rank is not None:
    #             return int(rank)
    #     # None to differentiate whether an environment variable was set at all
    #     return None

def set_gpu():
    # Check for GPU availability
    if tf.test.is_gpu_available():
        # Set GPU as the default device
        tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')
        # Enable memory growth for GPU
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)