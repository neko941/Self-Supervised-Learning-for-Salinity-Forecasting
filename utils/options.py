import argparse

from models.LTSF_Linear import *

model_dict = [
    {
        'model'  : LTSF_Linear__Tensorflow,
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_Linear__Tensorflow.yaml'
    },{
        'model'  : LTSF_NLinear__Tensorflow,
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_NLinear__Tensorflow.yaml'
    },{
        'model'  : LTSF_DLinear__Tensorflow,
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_DLinear__Tensorflow.yaml'
    },{
        'model'  : LTSF_NDLinearTime2VecRevIN__Tensorflow,
        'type'   : 'Tensorflow',
        'config' : r'.\configs\models\DeepLearning\LTSF_DLinear__Tensorflow.yaml'
    }
]

for model in model_dict:
    model.setdefault('alias', '')
    model.setdefault('help', '')
    model.setdefault('weight', None)

def parse_opt(ROOT, known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--batchsz', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--lag', type=int, default=5, help='')
    parser.add_argument('--ahead', type=int, default=1, help='')
    parser.add_argument('--offset', type=int, default=1, help='')
    parser.add_argument('--trainsz', type=float, default=0.7, help='')
    parser.add_argument('--valsz', type=float, default=0.1, help='')
    parser.add_argument('--resample', type=int, default=5, help='')
    parser.add_argument('--min_delta', type=float, default=0.00001, help='')

    parser.add_argument('--dataConfigs', action='append', help='dataset')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cyclicalPattern', action='store_true', help='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Nadam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Ftrl'], default='Adam', help='optimizer')
    parser.add_argument('--shuffle', type=str, choices=[None, 'local', 'global'], default=None, help='optimizer')
    parser.add_argument('--filling', type=str, choices=[None, 'forward', 'backward', 'min', 'max', 'mean'], default=None, help='')
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')
    parser.add_argument('--round', type=int, default=-1, help='Round decimals in results, -1 to disable')
    parser.add_argument('--individual', action='store_true', help='for LTSF Linear models')
    parser.add_argument('--debug', action='store_true', help='print debug information in table')
    parser.add_argument('--multimodels', action='store_true', help='split data of n segment ids for n models ')
    parser.add_argument('--workers', type=int, default=8, help='')
    parser.add_argument('--low_memory', action='store_true', help='Ultilize disk')
    parser.add_argument('--normalization', type=str, default=None, choices=[None, 'minmax', 'standard', 'robust'], help='')
    parser.add_argument('--ensemble', action='store_true', help='')

    parser.add_argument('--all', action='store_true', help='Use all available models')
    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--Tensorflow', action='store_true', help='')
    parser.add_argument('--Pytorch', action='store_true', help='')
    parser.add_argument('--LTSF', action='store_true', help='Using all LTSF Linear Models')
    parser.add_argument('--LTSF_RevIN', action='store_true', help='Using all LTSF Linear Models')

    for item in model_dict:
        parser.add_argument(f"--{item['model'].__name__}", action='store_true', help=f"{item['help']}")
        parser.add_argument(f"--{item['model'].__name__}_weight", type=str, default=None, help=f"Weight of {item['model'].__name__}")
        if item['alias'] != '': parser.add_argument(f"--{item['alias']}", action='store_true', help=f"{item['help']}")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def update_opt(opt):
    if opt.all:
        opt.MachineLearning = True
        opt.DeepLearning = True
    if opt.DeepLearning:
        opt.Tensorflow = True
        opt.Pytorch = True
    if opt.LTSF:
        opt.LTSF_Linear__Tensorflow = True
        opt.LTSF_NLinear__Tensorflow = True
        opt.LTSF_DLinear__Tensorflow = True
    for idx, model in enumerate(model_dict):
        model_dict[idx]['weight'] = vars(opt)[f'{model["model"].__name__}_weight']
        if model['alias'] != '':
            if vars(opt)[f'{model["alias"]}'] == True:
                vars(opt)[f'{model["model"].__name__}'] = True
                del vars(opt)[f'{model["alias"]}']
        if any([opt.Tensorflow and model['type']=='Tensorflow',
                opt.Pytorch and model['type']=='Pytorch',
                opt.MachineLearning and model['type']=='MachineLearning']): 
            vars(opt)[f'{model["model"].__name__}'] = True

    if opt.offset<opt.ahead: opt.offset=opt.ahead
    return opt