import os
import matplotlib.pyplot as plt

from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

def save_plot(filename, data, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    for datum in data: ax.plot(*datum['data'], color=datum['color'], label=datum['label'])
    ax.legend()
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    plt.title(os.path.basename(filename).split('.')[0])
    fig.savefig(filename)
    plt.close('all')

def progress_bar():
    return Progress("[bright_cyan][progress.description]{task.description}",
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("•Items"),
                    MofNCompleteColumn(), # "{task.completed}/{task.total}",
                    TextColumn("•Remaining"),
                    TimeRemainingColumn(),
                    TextColumn("•Total"),
                    TimeElapsedColumn())