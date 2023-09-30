import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import matplotlib
matplotlib.use('Agg') # Tcl_AsyncDelete: async handler deleted by the wrong thread

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # disable absl INFO and WARNING log messages

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation.

import gc
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt 
from utils.datasets import DatasetController
from utils.visuals import save_plot

from utils.general import set_seed
from utils.general import yaml_save
from utils.general import increment_path

from utils.options import parse_opt
from utils.options import update_opt
from utils.options import model_dict

from utils.metrics import metric_dict
from utils.rich2polars import table_to_df
from utils.activations import get_custom_activations

from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

from utils.general import convert_seconds
from utils.general import set_gpu
import csv
from utils.npy import NpyFileAppend


def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))

    """ Path to save configs """
    path_configs = Path(save_dir, 'configs')
    path_configs.mkdir(parents=True, exist_ok=True)

    """ Set seed """
    opt.seed = set_seed(opt.seed)

    """ Set GPU """
    set_gpu()

    """ Add custom function """
    get_custom_activations()

    """ Save init options """
    yaml_save(path_configs / 'opt.yaml', vars(opt))

    """ Update options """
    opt = update_opt(opt)

    """ To be dynamic """
    shuffle = False
    time_as_int = False
    enc_in = 1

    """ Save updated options """
    yaml_save(path_configs / 'updated_opt.yaml', vars(opt))

    """ Preprocessing dataset """
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    scaler = {}

    # total = 0
    assert len(opt.dataConfigs) > 0, 'There must be at least a dataset configuration to run'
    if len(opt.dataConfigs) == 1:
        # print((opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz))
        dataset = DatasetController(configsPath=opt.dataConfigs[0],
                                    resample=opt.resample,
                                    splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz),
                                    workers=opt.workers,
                                    lag=opt.lag, 
                                    ahead=opt.ahead, 
                                    offset=opt.offset,
                                    savePath=save_dir,
                                    filling=opt.filling,
                                    low_memory=opt.low_memory,
                                    normalization=opt.normalization,
                                    cyclicalPattern=opt.cyclicalPattern).execute()
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.GetData(shuffle=opt.shuffle!=None)
        scaler = dataset.scaler
        del dataset
        gc.collect()
        # print(f'{X_train.shape = }')
        # print(f'{y_train.shape = }')
        # print(f'{X_val.shape = }')
        # print(f'{y_val.shape = }')
        # print(f'{X_test.shape = }')
        # print(f'{y_test.shape = }')
    else:
        if opt.low_memory:
            path_values = Path(save_dir, 'values')
            path_values.mkdir(parents=True, exist_ok=True)
            save_file =  {
                'xtrain': {
                    'path': path_values / 'xtrain.npy',
                    'writer': NpyFileAppend(filename=path_values / 'xtrain.npy', delete_if_exists=True)
                },
                'ytrain': {
                    'path': path_values / 'ytrain.npy',
                    'writer': NpyFileAppend(filename=path_values / 'ytrain.npy', delete_if_exists=True)
                },
                'xval': {
                    'path': path_values / 'xval.npy',
                    'writer': NpyFileAppend(filename=path_values / 'xval.npy', delete_if_exists=True)
                },
                'yval': {    
                    'path': path_values / 'yval.npy',
                    'writer': NpyFileAppend(filename=path_values / 'yval.npy', delete_if_exists=True)
                },
                'xtest': {
                    'path': path_values / 'xtest.npy',
                    'writer': NpyFileAppend(filename=path_values / 'xtest.npy', delete_if_exists=True)
                },
                'ytest': {
                    'path': path_values / 'ytest.npy',
                    'writer': NpyFileAppend(filename=path_values / 'ytest.npy', delete_if_exists=True)
                }
            }
        for idx, cf in enumerate(opt.dataConfigs):
            print(json.dumps(cf, indent=4))
            skip_test = True if idx != 0 else False 
            dataset = DatasetController(configsPath=cf,
                                        resample=opt.resample,
                                        splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz),
                                        workers=opt.workers,
                                        lag=opt.lag, 
                                        ahead=opt.ahead, 
                                        offset=opt.offset,
                                        savePath=Path(save_dir, 'subdatasets', str(idx)),
                                        filling=opt.filling,
                                        low_memory=opt.low_memory,
                                        normalization=opt.normalization,
                                        cyclicalPattern=opt.cyclicalPattern,
                                        skip_test=skip_test).execute()
            xtrain, ytrain, xval, yval, xtest, ytest = dataset.GetData(shuffle=opt.shuffle=='local')
            # total += xtrain.shape[0]
            # print(f'{xtrain.shape = }')
            # print(f'{ytrain.shape = }')
            # print(f'{xval.shape = }')
            # print(f'{yval.shape = }')
            # print(f'{xtest.shape = }')
            # print(f'{ytest.shape = }')
            if idx == 0:
                scaler = dataset.scaler
                if opt.low_memory:
                    save_file['xtest']['writer'].append(xtest)
                    save_file['ytest']['writer'].append(ytest)
                else:
                    X_test = xtest
                    y_test = ytest
            else:
                if opt.low_memory:
                    # if ahead > 1: fortran_order=False 
                    # else: fortran_order=None 
                    np.save(Path(save_dir, 'temp.npy'), xtrain)
                    xtrain = np.load(Path(save_dir, 'temp.npy'), mmap_mode='r')
                    save_file['xtrain']['writer'].append(xtrain)
                    del(xtrain)
                    os.remove(Path(save_dir, 'temp.npy'))
                    save_file['ytrain']['writer'].append(ytrain)
                    save_file['xval']['writer'].append(xval)
                    save_file['yval']['writer'].append(yval)
                else:
                    X_train.extend(xtrain)
                    y_train.extend(ytrain)
                    X_val.extend(xval)
                    y_val.extend(yval)
            del dataset
            gc.collect() 
            print('=' * 50)
        if opt.low_memory:
            save_file['xtrain']['writer'].close()
            save_file['ytrain']['writer'].close()
            save_file['xval']['writer'].close()
            save_file['yval']['writer'].close()
            save_file['xtest']['writer'].close()
            save_file['ytest']['writer'].close()
            X_train = np.load(file=save_file['xtrain']['path'], mmap_mode='r+')
            y_train = np.load(file=save_file['ytrain']['path'], mmap_mode='r+')
            X_val = np.load(file=save_file['xval']['path'], mmap_mode='r')
            y_val = np.load(file=save_file['yval']['path'], mmap_mode='r')
            X_test = np.load(file=save_file['xtest']['path'], mmap_mode='r')
            y_test = np.load(file=save_file['ytest']['path'], mmap_mode='r')
        else:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)

    if opt.shuffle == 'global':
        X_train = np.random.shuffle(X_train)
        y_train = np.random.shuffle(y_train)
        X_val = np.random.shuffle(X_val)
        y_val = np.random.shuffle(y_val)
        X_test = np.random.shuffle(X_test)
        y_test = np.random.shuffle(y_test)

    print(f'{X_train.shape = }')
    # print(f'{total = }')
    print(f'{y_train.shape = }')
    print(f'{X_val.shape = }')
    print(f'{y_val.shape = }')
    print(f'{X_test.shape = }')
    print(f'{y_test.shape = }')
    # exit()

    """ Create result table """
    console = Console(record=True)
    table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
    [table.add_column(f'[green]{name}', justify='center') for name in ['Name', 'Time', *list(metric_dict.keys())]]


    if opt.ensemble: 
        r2s = []
        yhats = []

    """ Train models """
    for item in model_dict:
        if not vars(opt)[f'{item["model"].__name__}']: continue
        item['config'] = os.path.normpath(item['config'])
        print(item)
        shutil.copyfile(item['config'], path_configs/os.path.basename(item['config']))
        
        model = item['model'](input_shape=X_train.shape[-2:],
                              modelConfigs=item['config'], 
                              output_shape=opt.ahead, 
                              seed=opt.seed,
                              save_dir=save_dir,
                              enc_in=enc_in,
                              seq_len=X_train.shape[-1])
        model.build()
        if item['weight'] is not None: 
            model.built = True
            model.load(item['weight'])
        model.fit(patience=opt.patience, 
                  optimizer=opt.optimizer, 
                  loss=opt.loss, 
                  epochs=opt.epochs, 
                  learning_rate=opt.lr, 
                  batchsz=opt.batchsz,
                  X_train=X_train, y_train=y_train,
                  X_val=X_val, y_val=y_val,
                  time_as_int=time_as_int,
                  min_delta=opt.min_delta)
        model.save(file_name=f'{model.__class__.__name__}')

        print(f'{model.time_used = }')
        

        weight = model.best_weight
        if weight is None: weight = model.save(file_name=model.__class__.__name__)
        if weight is not None: model.load(weight)

        # predict values
        yhat = model.predict(X=X_test)
        if opt.ensemble: yhats.append(yhat)
        # print(f'{yhat.shape = }')
        ytrainhat = model.predict(X=X_train[:500])
        yvalhat = model.predict(X=X_val[:500])

        # calculate scores
        scores = model.score(y=y_test, 
                             yhat=yhat, 
                             r=opt.round, 
                             scaler=scaler)

        datum = [model.__class__.__name__, model.time_used, *scores]
        table.add_row(*datum)
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
        results = table_to_df(table)
        if opt.ensemble: r2s = results['R2'].to_list()
        results.write_csv(os.path.join(save_dir, 'results.csv'))
        # plot values
        # model.plot(save_dir=save_dir, y=y_train, yhat=ytrainhat, dataset='Train')
        # model.plot(save_dir=save_dir, y=y_val, yhat=yvalhat, dataset='Val')
        # model.plot(save_dir=save_dir, y=y_test, yhat=yhat, dataset='Test')
        
        # model.plot(save_dir=save_dir, y=y_train[:100], yhat=ytrainhat[:100], dataset='Train100')
        # model.plot(save_dir=save_dir, y=y_val[:100], yhat=yvalhat[:100], dataset='Val100')
        # model.plot(save_dir=save_dir, y=y_test[:100], yhat=yhat[:100], dataset='Test100')
    print(f'{X_train.shape = }')
    print(f'{y_train.shape = }')
    print(f'{X_val.shape = }')
    print(f'{y_val.shape = }')
    print(f'{X_test.shape = }')
    print(f'{y_test.shape = }')

    if opt.ensemble: 
        # print(len(r2s), len(yhats))
        # print(f'{r2s = }')
        from utils.metrics import score
        r2s = np.array([float(r2) for r2 in r2s])
        yhats = np.array(yhats)
        r2s = r2s / r2s.sum()
        # r2s = (r2s - r2s.min()) / (r2s.max() - r2s.min())
        # print(f'{r2s = }')
        # print(yhats.shape)
        yhats = np.array([a*b for a,b in zip(yhats, r2s)]).sum(axis=0)
        
        datum = ['Ensemble', '0s', *score(y=y_test, yhat=yhats, r=opt.round, scaler=scaler)]
        table.add_row(*datum)
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)  
        results = table_to_df(table)
        results.write_csv(os.path.join(save_dir, 'results.csv'))

def run(**kwargs):
    """ 
    Usage (example)
        import train
        train.run(all=True, 
                  configsPath=data.yaml,
                  lag=5,
                  ahead=1,
                  offset=1)
    """
    opt = parse_opt(ROOT=ROOT)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt(ROOT=ROOT)
    main(opt)